import onnx
import torch
import numpy as np
from onnx import helper
from onnx import onnx_pb as onnx_proto
from typing import Any
from ._base import CustomFunction
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


class ONNXOpFunction(CustomFunction):
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
            onnx.save_model(m, 'smoking.onnx')
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
    cls = type(_ox.get_unique_operator_type_name(op_type), (ONNXOpFunction, ),
               dict(
                   op_type=op_type,
                   opb_func=func,
                   attrs=attrs
               ))
    return cls.apply  # noqa


onnx_pad = create_op_function('Pad', _ox.pad)
onnx_where = create_op_function('Where', _ox.where)
onnx_greater = create_op_function('Greater', _ox.greater)
