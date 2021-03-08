import torch
import numpy as np
from onnx import onnx_pb as onnx_proto
from typing import List, Tuple, Optional, Union, Any, ContextManager, overload, Iterator, NamedTuple
from torch._C import long  # noqa
from torch.types import _int, _float, _bool, Number, _dtype, _device, _qscheme, _size, _layout  # noqa
from torch import strided, memory_format, contiguous_format, StringType  # noqa

from ._onnx_ops import ox as _ox
from ..eager_op import EagerOp


# class StringTensor(object):
#     def __init__(self, shape, value=None):
#         self._shape = shape
#         self._value = value

#     def __repr__(self):
#         return "StringTensor(shape={}, value={})".format(self._shape, self._value)


class _EagerTensor:

    def __init__(self, _t, name=None, sess=None, raw_data: Any = None):
        self._t = _t if isinstance(_t, torch.Tensor) else torch.tensor(_t)
        if isinstance(name, (tuple, list)):
            assert len(name) == 1, "Multiple names for one tensor!"
            name = name[0]
        self.name = '' if name is None else name
        self._ort_session = sess
        self.raw_data = raw_data

    def __repr__(self):
        if self.raw_data is not None:
            return "name: {}, \"{}\"".format(self.name, str(self.raw_data))
        else:
            return "name: {}, {}, dtype={}".format(self.name, repr(self._t), str(self._t.dtype))

    _NUMERIC_KINDS = set('buifc')
    _all_ops = {}

    @classmethod
    def is_numeric(cls, np_arr):
        return np_arr.dtype.kind in cls._NUMERIC_KINDS

    @classmethod
    def set_active_session(cls, sess):
        """
        set the active operator tracing log session. if sess is None, the active session will be removed
        :param sess:
        :return:
        """
        if not hasattr(cls, '_active_session'):
            cls._active_session = sess
            if sess is None:
                raise RuntimeError("unset the active session twice!")
        else:
            if sess is not None:
                raise RuntimeError("The active session already assigned!")
            delattr(cls, '_active_session')

    @classmethod
    def from_onnx(cls, raw_val, ort_sess, name):
        raw_data = None
        if cls.is_numeric(raw_val):
            val = torch.from_numpy(raw_val)
        else:
            # only keep the shape and value was stored by it-self.
            val = torch.empty(*raw_val.shape, dtype=torch.uint8)
            raw_data = raw_val
        t = cls(val, name, ort_sess, raw_data)
        return t

    @staticmethod
    def from_torch(_t, name=None):
        t_name = name if name is not None else "id_{}".format(id(_t))
        ts = _EagerTensor(_t, t_name)
        return ts

    def numpy(self):
        return self._t.numpy() if self.raw_data is None else self.raw_data

    @classmethod
    def get_trace_session(cls):
        if not hasattr(cls, '_active_session'):
            raise RuntimeError("the tracing not started yet!")
        return cls._active_session  # noqa

    def _to_binary_tensor_args(self, other):
        # convert self, other to [self, other], but if either is a number, convert that to a constant
        x, y = self, other
        if isinstance(y, (int, float, bool, np.ndarray)):
            y = self.from_torch(torch.tensor(y))
        elif isinstance(x, (int, float, bool, np.ndarray)):
            x = self.from_torch(torch.tensor(x))
        return x, y

    def __add__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.add(x0._t, x1._t)
        s = _ox.add(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __sub__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.sub(x0._t, x1._t)
        s = _ox.sub(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __mul__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.mul(x0._t, x1._t)
        s = _ox.mul(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __div__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.div(x0._t, x1._t)
        s = _ox.div(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __pow__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.pow(x0._t, x1._t)
        s = _ox.pow(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __matmul__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.matmul(x0._t, x1._t)
        s = _ox.matmul(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __lt__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.less(x0._t, x1._t)
        s = _ox.less(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __le__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.less_equal(x0._t, x1._t)
        s = _ox.less_equal(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __eq__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.equal(x0._t, x1._t)
        s = _ox.equal(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __ne__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.not_equal(x0._t, x1._t)
        s = _ox.not_equal(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __gt__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.greater(x0._t, x1._t)
        s = _ox.greater(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __ge__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.greater_equal(x0._t, x1._t)
        s = _ox.greater_equal(*_EagerTensor.ox_args(x0, x1))
        return self.from_torch(y, s)

    def __neg__(self):
        y = torch.neg([self])
        s = _ox.neg(*self.my_args())
        return self.from_torch(y, s)

    def __not__(self):
        y = torch.logical_not([self])
        s = _ox.not_op(*self.my_args())
        return self.from_torch(y, s)

    def __or__(self, other):
        y = torch.bitwise_or(*self._to_binary_tensor_args(other))
        s = _ox.or_op(*self.my_args())
        return self.from_torch(y, s)

    def __getitem__(self, indices):
        y = self.value.__getitem__(indices)

        # normalize indices to tuples of slices
        # Formats encountered:
        #  - a single int
        #  - a tuple of (int or slice)
        if not isinstance(indices, (tuple, list)):  # single item: make it a tuple
            indices = (indices,)
        squeeze = [axis for axis, index in enumerate(indices) if
                   isinstance(index, int)]  # which axes had a single index?
        indices = tuple(
            index if isinstance(index, slice) else slice(index, index + 1 if index != -1 else None, 1) for index in
            indices)  # make all tuple items of type Slice
        bs, es, ss, ds = [], [], [], []
        INT_MAX = 2 ** 63 - 1
        for axis, index in enumerate(indices):
            if not isinstance(index, slice):
                raise ValueError("Index expected")
            if index.start is None and index.stop is None:  # [:] can be skipped
                continue
            b, e, s = index.start, index.stop, index.step
            bs.append(b if b is not None else 0)
            es.append(e if e is not None else INT_MAX)
            ss.append(s if s is not None else 1)
            ds.append(axis)
        s = _ox.slice(*self.my_args(), starts=bs, ends=es, axes=ds, steps=ss)
        if squeeze:  # single index means we must drop the axis
            s = _ox.squeeze(*self.ox_name_args(s), axes=squeeze)

        return self.from_torch(y, s)

    @classmethod
    def ox_name_args(cls, input_names, output_names=None):
        """
        generate the arguments for ONNX model builder.
        :param input_names: input name list
        :param output_names: output name list, can be None, or [None]*output_n
        :return: input_names, output_names, container, operator_name
        """
        container = cls.get_trace_session().container
        if output_names is None:
            output_names = [None]  # by default, there is only one output

        output_names = [_ox.get_unique_tensor_name(str(n_))
                        if output_names[n_] is None else
                        output_names[n_] for n_ in range(len(output_names))]
        operator_name = None
        return input_names, output_names, container, operator_name

    @classmethod
    def ox_args(cls, tensors, output_names=None):
        input_names = [ts_.name for ts_ in tensors]
        return cls.ox_name_args(input_names, output_names)

    def my_args(self):
        return self.ox_args([self])

    def __getattribute__(self, attr):
        """
        A little hack that allows to call unary operators in a chaining fashion,
        e.g. x.shape() instead of ox.shape(x).
        """
        if attr in self._all_ops:
            f = self._all_ops
            return f
        else:
            return object.__getattribute__(self, attr)

    @staticmethod
    def normalize_seq(list_or_tuple):
        return [x.value.item() if isinstance(x, _EagerTensor) else x for x in list_or_tuple]

    @property
    def value(self) -> Union[torch.Tensor, Any]:
        return self.raw_data if self.raw_data else self._t

    def long(self):
        y = self._t.long()
        s = _ox.cast(*self.my_args(), to=onnx_proto.TensorProto.INT64)
        return self.from_torch(y, s[0])

    def cumsum(self, dim: _int, *, dtype: Optional[_dtype] = None):
        y = self._t.cumsum(dim, dtype=dtype)
        s = _ox.cumsum(*self.my_args())
        return self.from_torch(y, s[0])

    def size(self):
        y = self._t.size()
        s = _ox.shape(*self.my_args())
        return self.from_torch(y, s)

    def type(self, dtype: Union[str, _dtype], non_blocking: _bool=False):
        y = self._t.type(dtype. non_blocking)
        ty_dict = {torch.float32: onnx_proto.TensorProto.FLOAT,
                   torch.long: onnx_proto.TensorProto.INT64}
        s = _ox.cast(*self.my_args(), to=ty_dict[dtype])
        return self.from_torch(y, s)

    def to(self, device):
        y = self._t.to(device)
        s = _ox.identity(*self.my_args())
        return self.from_torch(y, s[0])

    def clone(self):
        y = self._t.clone()
        s = _ox.identity(*self.my_args())
        return self.from_torch(y, s[0])

    def masked_fill(self, mask, value):
        return self.from_torch(self._t.masked_fill(mask.value, value))

    def unsqueeze(self, dim: _int):
        y = self._t.unsqueeze(dim)
        s = _ox.unsqueeze(*self.my_args(), dim)
        return self.from_torch(y, s[0])

    def squeeze(self, dim: _int):
        y = self._t.squeeze(dim)
        s = _ox.squeeze(*self.my_args(), dim)
        return self.from_torch(y, s[0])


def tensor(data: Any, dtype: Optional[_dtype] = None, device: Union[_device, str, None] = None,
           requires_grad: _bool = False) -> _EagerTensor:  # noqa

    y = torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
    s = _ox.constant([], [_ox.get_unique_tensor_name('const')], _EagerTensor.get_trace_session().container, None, value=data)
    return _EagerTensor.from_torch(y, s)


def empty(*size: _int, memory_format: Optional[memory_format] = None, out: Optional[_EagerTensor] = None,
          dtype: _dtype = None, layout: _layout = strided, device: Union[_device, str, None] = None,
          requires_grad: _bool = False) -> _EagerTensor:  # noqa
    n_size = _EagerTensor.normalize_seq(size)
    y = torch.empty(*n_size, memory_format=memory_format, out=out,
                    dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    
    container = _EagerTensor.get_trace_session().container
    con_x = [] 
    if isinstance(size, (list, tuple)):
        for x in size:
            if not isinstance(x, _EagerTensor):
                x_h = _ox.constant_of_shape([], [_ox.get_unique_tensor_name('const')], container, None, value=x)
            else:
                x_h = x
            con_x.append(x_h)

    allnames = _ox.concat(con_x, [_ox.get_unique_tensor_name('concat')], container, None, value=x)
    s = _ox.constant_of_shape(allnames,[_ox.get_unique_tensor_name('concat')], container, None)
    return _EagerTensor.from_torch(y, s)


def zeros(*size: _int, out: Optional[_EagerTensor] = None, dtype: _dtype = None, layout: _layout = strided,
          device: Union[_device, str, None] = None, requires_grad: _bool = False) -> _EagerTensor:  # noqa
    n_size = _EagerTensor.normalize_seq(size)
    y = torch.zeros(*n_size, out=out, dtype=dtype,
                                         layout=layout, device=device, requires_grad=requires_grad)

    container = _EagerTensor.get_trace_session().container
    con_x = [] 
    if isinstance(size, (list, tuple)):
        for x in size:
            if not isinstance(x, _EagerTensor):
                x_h = _ox.constant_of_shape([], [_ox.get_unique_tensor_name('const')], container, None, value=x)
            else:
                x_h = x
            con_x.append(x_h)

    allnames = _ox.concat(con_x, [_ox.get_unique_tensor_name('concat')], container, None)
    s = _ox.constant_of_shape(allnames,[_ox.get_unique_tensor_name('concat')], container, None, value=0)
    return _EagerTensor.from_torch(y, s)


def argmax(input_ts: _EagerTensor, dim: Optional[_int] = None, keepdim: _bool = False) -> _EagerTensor:  # noqa
    y = torch.argmax(input_ts.value, dim, keepdim)
    s = _ox.argmax(*input_ts.my_args(), axis=dim, keepdims=keepdim)
    return _EagerTensor.from_torch(y, s)


def cat(tensors: Union[Tuple[_EagerTensor, ...], List[_EagerTensor]], dim, *, out: Optional[_EagerTensor] = None) -> _EagerTensor:  # noqa
    res = torch.cat([t_.value for t_ in tensors], dim, out=out)
    oname = _ox.concat(*_EagerTensor.ox_args(tensors), dim)
    return _EagerTensor.from_torch(res, oname[0])


def onnx_loop(*tensors: Union[_EagerTensor, int]):
    res = _EagerTensor.normalize_seq(tensors)
    return range(res[0])


class _TracingEagerOp(EagerOp):
    def __call__(self, *args, **kwargs):
        np_args = [ts_.numpy() if isinstance(ts_, _EagerTensor) else ts_ for ts_ in args]
        outseq = super().__call__(*np_args, **kwargs)

        outputs = [_EagerTensor.from_onnx(outseq[n_], self.ort_session, out_.name
                                    ) for n_, out_ in enumerate(self.ort_session.get_outputs())]
        # FIXME: The tokenizer support attention_mask output
        if self.onnx_model.graph.name.startswith('og_GPT2Tokenizer'):
            outputs.append(_EagerTensor.from_torch(torch.tensor(
                [[1] * outputs[0].value.shape[1]], dtype=torch.float32), 'attention_mask'))

        y_names = [y.name for y in outputs]
        _ox.model_call(*_EagerTensor.ox_args(args, output_names=y_names), oxml=self.onnx_model)
        return tuple(outputs)


def op_from_customop(op_type, *args, **kwargs) -> _TracingEagerOp:
    return _TracingEagerOp.from_customop(op_type, *args, **kwargs)


def op_from_model(path_or_model, *args, **kwargs) -> _TracingEagerOp:
    return _TracingEagerOp.from_model(path_or_model, *args, **kwargs)


_EagerTensor._all_ops = {'argmax': argmax}
tensor_from_onnx = _EagerTensor.from_onnx
tensor_set_session = _EagerTensor.set_active_session
