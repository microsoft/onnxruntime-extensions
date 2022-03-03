import onnx
import torch
import numpy as np
from typing import Any, List
from onnx import helper
from onnx import onnx_pb as onnx_proto
from torch.onnx import register_custom_op_symbolic

from ._base import CustomFunction, ProcessingModule, ProcessingScriptModule
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


# @torch.jit.ignore(drop=True)
# def _pass_through_call(*args, **kwargs):
#     model_id = args[0]
#     path_or_model = OnnxModelFunction._id_path_map[model_id]
#     func = OrtPyFunction.from_model(path_or_model)
#     return torch.from_numpy(
#         func(*list(_i.numpy() if isinstance(_i, torch.Tensor) else _i for _i in args[1:]), **kwargs))
#
#
# @torch.jit.ignore
# class OnnxModelFunction:
#     """
#     This class turns an ONNX model to be Python function, which can be used in jit.script
#     """
#     _id_path_map = {}    # cannot use the string directly since jit.script doesn't support the data type
#     _model_id_counter = 0
#     str_model_function_id = '_model_function_id'
#     str_model_id = '_model_id'
#     str_model_attached = '_model_attached'
#
#     @classmethod
#     def symbolic(cls, g, *inputs):
#         return g.op(
#             "ai.onnx.contrib::{}{}".format(
#                 OnnxModelFunction.str_model_function_id, cls.get_id()), *inputs)
#
#     @classmethod
#     def apply(cls: Any, *args: List[torch.Tensor], **kwargs: Any) -> List[torch.Tensor]:
#         return _pass_through_call(cls.get_id(), *args, **kwargs)
#
#     @classmethod
#     def get_id(cls):
#         if not hasattr(cls, OnnxModelFunction.str_model_id):
#             model_or_path = getattr(cls, OnnxModelFunction.str_model_attached)
#             _id = id(model_or_path)
#             setattr(cls, OnnxModelFunction.str_model_id, _id)
#             OnnxModelFunction._id_path_map[_id] = model_or_path
#         return getattr(cls, OnnxModelFunction.str_model_id)
#

# class _OnnxModelModule(torch.nn.Module):
#     def __init__(self, model_or_path):
#         super(_OnnxModelModule, self).__init__()
#         # self.fn_model = type("_Mfunc_{}".format(id(model_or_path)), (OnnxModelFunction,), {
#         #     OnnxModelFunction.str_model_attached: model_or_path}).apply
#         _OnnxModelModule.fn_model = OrtPyFunction.from_model(model_or_path)
#
#     def symbolic(cls, g, *inputs):
#         return g.op(
#             "ai.onnx.contrib::{}{}".format(
#                 OnnxModelFunction.str_model_function_id, cls.get_id()), *inputs)
#
#     @staticmethod
#     @torch.jit.ignore
#     def _pass_through_call(*args, **kwargs):
#         func = _OnnxModelModule.fn_model
#         return torch.from_numpy(
#             func(*list(_i.numpy() if isinstance(_i, torch.Tensor) else _i for _i in args[0:]), **kwargs))
#
#     def forward(self, args):
#         # if args[0].nelement() == 0:
#         #     return args
#         arg1 = args + torch.tensor(1.0) - torch.tensor(1.0)
#         # return _OnnxModelModule._pass_through_call(arg1)
#         return  self.fn_model(arg1)

#
# def create_model_function(model_or_path):
#     _M = type("_Mfunc_{}".format(id(model_or_path)), (OnnxModelFunction, ), {
#          OnnxModelFunction.str_model_attached: model_or_path})
#     return _M

# def create_model_function(model_or_path, example_inputs=None):
#     model = _OnnxModelModule(model_or_path)
#     return model
#     # return torch.jit.trace_module(model, {'forward': torch.rand(1, 3, 224, 224)}, check_trace=False, check_tolerance=1e-7)

class _OnnxModelFunction:
    id_object_map = {}    # cannot use the string directly since jit.script doesn't support the data type
    str_model_function_id = '_model_function_id'
    str_model_id = '_model_id'
    str_model_attached = '_model_attached'


@torch.jit.ignore
def invoke_onnx_model(model_id: int, *args, **kwargs):
    model_or_path = _OnnxModelFunction.id_object_map.get(model_id)
    if model_or_path is None:
        raise ValueError("cannot find id={} registered!".format(model_id))
    func = OrtPyFunction.from_model(model_or_path)
    return torch.from_numpy(
        func(*list(_i.numpy() if isinstance(_i, torch.Tensor) else _i for _i in args), **kwargs))


class _OnnxTracedFunction(CustomFunction):
    @classmethod
    def forward(cls, ctx: Any, *args: Any, **kwargs: Any) -> Any:
        return invoke_onnx_model(args[0].item(), *args[1:], **kwargs)

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


class SequenceProcessingModule(ProcessingModule):
    def __init__(self, mod1, mod2):
        super(SequenceProcessingModule, self).__init__()
        self.model1 = mod1
        self.model2 = mod2
        self.model1_function_id = 0
        self.model2_function_id = 0
        if isinstance(mod1, onnx.ModelProto):
            self.model1_function_id = create_model_function(self.model1)
        if isinstance(mod2, onnx.ModelProto):
            self.model2_function_id = create_model_function(self.model2)

    def forward(self, *args):
        if self.model1_function_id != 0:
            step1 = _OnnxTracedFunction.apply(torch.tensor(self.model1_function_id), *args)
        else:
            step1 = self.model1.forward(*args)

        if self.model2_function_id != 0:
            outputs = _OnnxTracedFunction.apply(torch.tensor(self.model2_function_id), step1)
        else:
            outputs = self.model2.forward(step1)

        return outputs


def _symbolic_pythonop(g: torch._C.Graph, n: torch._C.Node, *args, **kwargs):
    name = kwargs["name"]
    if name == invoke_onnx_model.__name__:
        # id = torch.onnx.symbolic_helper._maybe_get_scalar(args[0]).item()
        ret = g.op("ai.onnx.contrib::_ModelFunctionCall", *args)
    else:
        # Logs a warning and returns None
        import warnings
        return warnings.warn("prim::PythonOp", "unknown node kind: " + name)
    # Copy type and shape from original node.
    ret.setType(args[-1].type())
    return ret


register_custom_op_symbolic("prim::PythonOp", _symbolic_pythonop, 1)
