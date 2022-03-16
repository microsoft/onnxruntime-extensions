import onnx
import torch
import numpy as np
from typing import Any
from onnx import helper
from onnx import onnx_pb as onnx_proto
from distutils.version import LooseVersion
from torch.onnx import register_custom_op_symbolic

from ._base import CustomFunction, ProcessingTracedModule
from ._onnx_ops import ox as _ox, schema as _schema
from ._onnx_ops import ONNXElementContainer, make_model_ex
from .._ortapi2 import OrtPyFunction, get_opset_version_from_ort


def _is_numpy_object(x):
    return isinstance(x, (np.ndarray, np.generic))


def _is_numpy_string_type(arr):
    return arr.dtype.kind in {'U', 'S'}


def _is_string_type(x):
    if not _is_numpy_object(x):
        x = np.array(x)
    return _is_numpy_string_type(x)


def _to_onnx_type(dtype):
    ty_dict = {torch.bool: onnx_proto.TensorProto.BOOL,
               torch.float32: onnx_proto.TensorProto.FLOAT,
               torch.float64: onnx_proto.TensorProto.DOUBLE,
               torch.long: onnx_proto.TensorProto.INT64,
               torch.int32: onnx_proto.TensorProto.INT32}
    # ...
    return ty_dict.get(dtype, onnx_proto.TensorProto.STRING)


class OnnxOpFunction(CustomFunction):
    @classmethod
    def get_next_id_name(cls, name_base):
        name = 'cls' if name_base is None else name_base
        _cid = getattr(cls, '_cid', 1)
        cls._cid = _cid + 1
        return "{}_{}".format(name, _cid)

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs

    @classmethod
    def build_model(cls, opset_version, *args):
        # build the one node graph
        if isinstance(args[0], list):
            args = [np.asarray(_i) for _i in args]
        ec = ONNXElementContainer(get_opset_version_from_ort() if opset_version is None else opset_version)
        attrs = cls.attrs
        vi_inputs = [helper.make_tensor_value_info(
            'it_' + str(id(_arg)), _to_onnx_type(_arg.dtype), list(_arg.shape))
            for _arg in args]
        inputs = [_vi.name for _vi in vi_inputs]
        if hasattr(cls.opb_func, 'outputs') and len(cls.opb_func.outputs) > 0:
            vi_outputs = [helper.make_tensor_value_info(
                cls.get_next_id_name('ot'), *_schm) for _schm in cls.opb_func.outputs]
        else:
            vi_outputs = [helper.make_tensor_value_info(
                cls.get_next_id_name('ot'), onnx_proto.TensorProto.FLOAT, []
            )]
        outputs = [_vi.name for _vi in vi_outputs]
        # build the node
        opfunc = cls.opb_func
        opfunc(inputs, outputs, ec, None, **attrs)
        g = helper.make_graph(ec.nodes, cls.get_next_id_name('g'), vi_inputs, vi_outputs)
        m = make_model_ex(g, ec.node_domain_version_pair_sets, ec.target_opset)
        return m

    @classmethod
    @torch.jit.unused
    def _onnx_call(cls, ctx, *args) -> Any:
        m = cls.build_model(None, *args)
        try:
            f = OrtPyFunction.from_model(m)
            result = f(*list(_i.numpy() if isinstance(_i, torch.Tensor) else _i for _i in args))
        except Exception as e:
            onnx.save_model(m, '_temp_debugging.onnx')
            raise e

        results = result if isinstance(result, tuple) else [result]
        return tuple([torch.from_numpy(_o) for _o in results]) if len(results) > 1 else torch.from_numpy(results[0])

    @classmethod
    def forward(cls, ctx: Any, *args: Any, **kwargs: Any) -> Any:
        return cls._onnx_call(ctx, *args, **kwargs)

    @classmethod
    def symbolic(cls, g, *args):
        return g.op(cls.op_type, *args)


def create_op_function(op_type: str, func, **attrs):
    if _ox.is_raw(func):
        func = _schema(func.__func__)
    cls = type(_ox.get_unique_operator_type_name(op_type), (OnnxOpFunction, ),
               dict(
                   op_type=op_type,
                   opb_func=func,
                   attrs=attrs
               ))
    return cls.apply  # noqa


onnx_pad = create_op_function('Pad', _ox.pad)
onnx_where = create_op_function('Where', _ox.where)
onnx_greater = create_op_function('Greater', _ox.greater)


class PythonOpFunction:
    """
    PythonOpFunction wraps a generic Python function which skips forward operation on torch.onnx.exporter
    converting in the script mode, since the exporter cannot support the APIs from external package, like Numpy.
    BTW, Autograd.Function cannot be used torch.jit.script.
    """
    id_func_map = {}
    current_func_id = 0

    @staticmethod
    def _get_next_id():
        PythonOpFunction.current_func_id += 1
        return PythonOpFunction.current_func_id

    @staticmethod
    @torch.jit.ignore
    def _pass_through_call(*args, **kwargs):
        func_id = args[0]
        func = PythonOpFunction.id_func_map[func_id]
        return torch.from_numpy(func.forward(*args[1:], **kwargs))

    @classmethod
    def apply(cls, *args, **kwargs):
        return PythonOpFunction._pass_through_call(cls.get_id(), *args, **kwargs)

    @classmethod
    def get_id(cls):
        if not hasattr(cls, 'func_id'):
            _id = PythonOpFunction._get_next_id()
            setattr(cls, 'func_id', _id)
            PythonOpFunction.id_func_map[_id] = cls
        return cls.func_id


class _OnnxModelFunction:
    id_object_map = {}    # cannot use the string directly since jit.script doesn't support the data type
    str_model_function_id = '_model_function_id'
    str_model_id = '_model_id'
    str_model_attached = '_model_attached'


@torch.jit.ignore
def _invoke_onnx_model(model_id: int, *args, **kwargs):
    model_or_path = _OnnxModelFunction.id_object_map.get(model_id)
    if model_or_path is None:
        raise ValueError("cannot find id={} registered!".format(model_id))
    func = OrtPyFunction.from_model(model_or_path)
    return torch.from_numpy(
        func(*list(_i.numpy() if isinstance(_i, torch.Tensor) else _i for _i in args), **kwargs))


@torch.jit.ignore
def invoke_onnx_model1(model_id: int, arg0):
    return _invoke_onnx_model(model_id, arg0)


@torch.jit.ignore
def invoke_onnx_model2(model_id: int, arg0, arg1):
    return _invoke_onnx_model(model_id, arg0, arg1)


@torch.jit.ignore
def invoke_onnx_model3(model_id: int, arg0, arg1, arg2):
    return _invoke_onnx_model(model_id, arg0, arg1, arg2)


class _OnnxTracedFunction(CustomFunction):
    @classmethod
    def forward(cls, ctx: Any, *args: Any, **kwargs: Any) -> Any:
        return _invoke_onnx_model(args[0].item(), *args[1:], **kwargs)

    @classmethod
    def symbolic(cls, g, *args):
        return g.op('ai.onnx.contrib::_ModelFunctionCall', *args)


def create_model_function(model_or_path):
    _id = id(model_or_path)
    assert _id != 0, "internal error: the id of a Python object is 0."
    _OnnxModelFunction.id_object_map[_id] = model_or_path
    return _id


def get_id_models():
    return _OnnxModelFunction.id_object_map


class SequenceProcessingModule(ProcessingTracedModule):
    def __init__(self, *models):
        super(SequenceProcessingModule, self).__init__()
        self.model_list = models
        self.model_function_ids = []
        for mdl_ in models:
            if isinstance(mdl_, onnx.ModelProto):
                self.model_function_ids.append(create_model_function(mdl_))
            else:
                self.model_function_ids.append(0)

    def forward(self, *args):
        outputs = args
        for idx_, mdl_ in enumerate(self.model_list):
            if not isinstance(outputs, tuple):
                outputs = (outputs, )
            if self.model_function_ids[idx_] != 0:
                outputs = _OnnxTracedFunction.apply(torch.tensor(self.model_function_ids[idx_]), *outputs)
            else:
                outputs = self.model_list[idx_].forward(*outputs)

        return outputs


def _symbolic_pythonop(g: torch._C.Graph, n: torch._C.Node, *args, **kwargs):
    name = kwargs["name"]
    if name.startswith(invoke_onnx_model1.__name__[:-1]):
        # id = torch.onnx.symbolic_helper._maybe_get_scalar(args[0]).item()
        ret = g.op("ai.onnx.contrib::_ModelFunctionCall", *args)
    else:
        # Logs a warning and returns None
        import warnings
        return warnings.warn("prim::PythonOp", "unknown node kind: " + name)
    # Copy type and shape from original node.
    ret.setType(n.output().type())
    return ret


if LooseVersion(torch.__version__) >= LooseVersion("1.11"):
    register_custom_op_symbolic("prim::PythonOp", _symbolic_pythonop, 1)
