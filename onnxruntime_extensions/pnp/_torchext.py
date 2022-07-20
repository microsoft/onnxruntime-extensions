import onnx
import torch
import numpy as np
from typing import Any
from onnx import helper
from onnx import onnx_pb as onnx_proto
from distutils.version import LooseVersion
from torch.onnx import register_custom_op_symbolic

from ._utils import ONNXModelUtils
from ._base import CustomFunction, ProcessingTracedModule, is_processing_module
from ._onnx_ops import ox as _ox, schema as _schema
from ._onnx_ops import ONNXElementContainer, make_model_ex
from .._ortapi2 import OrtPyFunction, get_opset_version_from_ort


def _is_numpy_object(x):
    return isinstance(x, (np.ndarray, np.generic))


def _is_numpy_string_type(arr):
    return arr.dtype.kind in {'U', 'S'}


def _is_string_type(x):
    if isinstance(x, list):
        return any(_is_string_type(e) for e in x)
    elif isinstance(x, torch.Tensor):
        return False
    elif not _is_numpy_object(x):
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
    cls = type(_ox.get_unique_operator_type_name(op_type), (OnnxOpFunction,),
               dict(
                   op_type=op_type,
                   opb_func=func,
                   attrs=attrs
               ))
    return cls.apply  # noqa


onnx_pad = create_op_function('Pad', _ox.pad)
onnx_where = create_op_function('Where', _ox.where)
onnx_greater = create_op_function('Greater', _ox.greater)


class _OnnxModelFunction:
    id_object_map = {}  # cannot use the string directly since jit.script doesn't support the data type
    id_function_map = {}
    str_model_function_id = '_model_function_id'
    str_model_id = '_model_id'
    str_model_attached = '_model_attached'


@torch.jit.ignore
def _invoke_onnx_model(model_id: int, *args, **kwargs):
    func = _OnnxModelFunction.id_function_map.get(model_id, None)
    if not func:
        model_or_path = _OnnxModelFunction.id_object_map.get(model_id)
        if model_or_path is None:
            raise ValueError("cannot find id={} registered!".format(model_id))
        func = OrtPyFunction.from_model(model_or_path)
        _OnnxModelFunction.id_function_map[model_id] = func
    results = func(*list(_i.numpy() if isinstance(_i, torch.Tensor) else _i for _i in args), **kwargs)
    return tuple(
        [torch.from_numpy(_o) for _o in results]) if isinstance(results, tuple) else torch.from_numpy(results)


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
        ret = g.op('ai.onnx.contrib::_ModelFunctionCall', *args)
        model_id = torch.onnx.symbolic_helper._maybe_get_scalar(args[0])  # noqa
        if not model_id:
            return ret

        func = _OnnxModelFunction.id_function_map.get(model_id.item(), None)
        if not func or len(func.outputs) <= 1:
            return ret

        outputs = [ret]
        for _ in range(len(func.outputs) - 1):
            outputs.append(ret.node().addOutput())

        return tuple(outputs)


def create_model_function(model_or_path):
    _id = id(model_or_path)
    assert _id != 0, "internal error: the id of a Python object is 0."
    _OnnxModelFunction.id_object_map[_id] = model_or_path
    return _id


def get_id_models():
    return _OnnxModelFunction.id_object_map


class OnnxTracedModelFunction:
    def __init__(self, onnx_model):
        self.func_id = create_model_function(onnx_model)

    def __call__(self, *args, **kwargs):
        return _OnnxTracedFunction.apply(torch.tensor(self.func_id), *args, **kwargs)


class _OnnxModelModule(torch.nn.Module):
    def __init__(self, mdl):
        super(_OnnxModelModule, self).__init__()
        self.function = OnnxTracedModelFunction(mdl)

    def forward(self, *args):
        return self.function(*args)


def _symbolic_pythonop(g: torch._C.Graph, n: torch._C.Node, *args, **kwargs):
    name = kwargs["name"]
    if name.startswith(invoke_onnx_model1.__name__[:-1]):
        # NB: if you want to get the value of the first argument, i.e. the model id,
        # you can get it by torch.onnx.symbolic_helper._maybe_get_scalar(args[0]).item()
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


class SequentialProcessingModule(ProcessingTracedModule):
    def __init__(self, *models):
        super(SequentialProcessingModule, self).__init__()
        self.model_list = torch.nn.ModuleList()
        for mdl_ in models:
            if isinstance(mdl_, onnx.ModelProto):
                self.model_list.append(_OnnxModelModule(mdl_))
            elif is_processing_module(mdl_):
                self.model_list.append(mdl_)
            else:
                assert callable(mdl_), "the model type is not recognizable."
                self.model_list.append(ProcessingTracedModule(mdl_))

    def forward(self, *args):
        outputs = args
        with torch.no_grad():
            for idx_, mdl_ in enumerate(self.model_list):
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
                outputs = mdl_(*outputs)

        return outputs

    def export(self, *args, **kwargs):
        prefix_m = None
        core_m = self
        raw_input_flag = any(_is_string_type(x_) for x_ in args)
        if raw_input_flag:
            # NB: torch.onnx.export doesn't support exporting a module accepting string type input,
            # So, in this case, the module will be separated into two parts to use the customized export.
            m0 = self.model_list[0]
            new_args = m0(*args)
            if not isinstance(new_args, tuple):
                new_args = (new_args, )
            prefix_m = m0.export(*args, **kwargs)
            args = new_args
            core_m = SequentialProcessingModule(*self.model_list[1:])
        if prefix_m is None:
            return super().export(*args, **kwargs)
        else:
            oxml = core_m.export(*args, **kwargs)
            model = ONNXModelUtils.join_models(prefix_m, oxml)

            # Rename the input/output node names if the user has provided any substitutions!
            # Ref: https://github.com/onnx/onnx/issues/2052
            # Known issue: This logic doesn't deal with subgraphs.
            if (('input_names' in kwargs) or ('output_names' in kwargs)) and \
              (kwargs['input_names'] or kwargs['output_names']):
                swaps = {}
                if 'input_names' in kwargs and kwargs['input_names']:
                    assert len(model.graph.input) == len(kwargs['input_names']), \
                      "Expecting {} input names but got {}".format(
                        len(model.graph.input), len(kwargs['input_names']))
                    for n, new_name in zip(model.graph.input, kwargs['input_names']):
                        swaps[n.name] = new_name
                        n.name = new_name

                if 'output_names' in kwargs and kwargs['output_names']:
                    assert len(model.graph.output) == len(kwargs['output_names']), \
                      "Expecting {} output names but got {}".format(
                        len(model.graph.output), len(kwargs['output_names']))
                    for n, new_name in zip(model.graph.output, kwargs['output_names']):
                        swaps[n.name] = new_name
                        n.name = new_name

                if swaps:
                    for n in model.graph.node:
                        for j in range(len(n.input)):
                            n.input[j] = swaps.get(n.input[j], n.input[j])

                        for j in range(len(n.output)):
                            n.output[j] = swaps.get(n.output[j], n.output[j])

                    for n in model.graph.initializer:
                        n.name = swaps.get(n.name, n.name)

            return model
