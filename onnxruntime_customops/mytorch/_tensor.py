import torch
import numpy as np
from typing import List, Tuple, Optional, Union, Any, ContextManager, overload, Iterator, NamedTuple  # noqa
from torch.types import _int, _float, _bool, Number, _dtype, _device, _qscheme, _size, _layout  # noqa
from torch import strided, memory_format, contiguous_format  # noqa

from ._onnx_ops import ox as _ox
from ..eager_op import EagerOp


class StringType:
    is_complex = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    is_floating_point = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default

    is_signed = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default


string = StringType()


class StringTensor(object):
    def __init__(self, shape, value=None):
        self._shape = shape
        self._value = value

    def __repr__(self):
        return "StringTensor(shape={}, value={})".format(self._shape, self._value)


class Tensor:
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
    def from_onnx(cls, ort_sess, out_val, name):
        val = torch.from_numpy(out_val)
        t = Tensor(val)
        t._ort_session = ort_sess
        t.name = name
        return t

    @staticmethod
    def from_torch(_t, name=None):
        t_name = name if name is not None else "id_{}".format(id(_t))
        ts = Tensor(_t, t_name)
        return ts

    def __init__(self, _t, name=None):
        # FIXME: the default is int64?
        self._t = _t if isinstance(_t, torch.Tensor) else torch.tensor(_t, dtype=torch.int64)
        self.name = '' if name is None else name

    def __repr__(self):
        return "name: {}, {}".format(self.name, repr(self._t))

    @classmethod
    def get_trace_session(cls):
        if not hasattr(cls, '_active_session'):
            raise RuntimeError("the tracing not started yet!")
        return cls._active_session  # noqa

    @classmethod
    def _process_inputs(self, inputs, name):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        ox_inputs = []
        for i_ in inputs:
            ox_n = i_
            if isinstance(i_, np.ndarray):
                ox_n = self._scope.get_unique_variable_name(name + '_i')
                self._container.add_initializer(
                    ox_n,
                    NP_TYPE_TO_TENSOR_TYPE[i_.dtype],
                    i_.shape,
                    i_.flatten()
                )
            elif isinstance(i_, (tuple, list)):
                ox_n = self._scope.get_unique_variable_name(name + i_[0])
                self._container.add_initializer(
                    ox_n,
                    i_[1],
                    i_[2].shape,
                    i_[2].flatten()
                )
            elif isinstance(ox_n, str):
                pass
            else:
                raise ValueError(
                    'Unknown type for ONNX initializer: {}'.format(type(ox_n)))
            ox_inputs.append(ox_n)

        return ox_inputs

    def _process_outputs(self, outputs, name):
        if outputs is None:
            ox_outputs = 1
        else:
            ox_outputs = outputs
        if isinstance(ox_outputs, int):
            ox_outputs = [self._scope.get_unique_variable_name(
                name + str(i_)) for i_ in range(ox_outputs)]
        elif isinstance(ox_outputs, (list, tuple)):
            pass
        else:
            raise ValueError(
                'Unknown type for outputs: {}'.format(type(ox_outputs)))
        return ox_outputs

    def _to_binary_tensor_args(self, other):
        # convert self, other to [self, other], but if either is a number, convert that to a constant
        x, y = self, other
        if isinstance(y, (int, float, bool, np.ndarray)):
            y = self.from_torch(torch.tensor(y))
        elif isinstance(x, (int, float, bool, np.ndarray)):
            x = self.from_torch(torch.tensor(x))
        return x.value, y.value

    def __add__(self, other):
        return self.from_torch(torch.add(*self._to_binary_tensor_args(other)))

    def __sub__(self, other):
        return self.from_torch(torch.sub(*self._to_binary_tensor_args(other)))

    def __mul__(self, other):
        return self.from_torch(torch.mul(*self._to_binary_tensor_args(other)))

    def __div__(self, other):
        return self.from_torch(torch.div(*self._to_binary_tensor_args(other)))

    def __pow__(self, other):
        return self.from_torch(torch.pow(*self._to_binary_tensor_args(other)))

    def __matmul__(self, other):
        return self.from_torch(torch.matmul(*self._to_binary_tensor_args(other)))

    def __lt__(self, other):
        return self.from_torch(torch.less(*self._to_binary_tensor_args(other)))

    def __le__(self, other):
        return self.from_torch(torch.less_equal(*self._to_binary_tensor_args(other)))

    def __eq__(self, other):
        return self.from_torch(torch.equal(*self._to_binary_tensor_args(other)))

    def __ne__(self, other):
        return self.from_torch(torch.not_equal(*self._to_binary_tensor_args(other)))

    def __gt__(self, other):
        return self.from_torch(torch.greater(*self._to_binary_tensor_args(other)))

    def __ge__(self, other):
        return self.from_torch(torch.greater_equal(*self._to_binary_tensor_args(other)))

    def __neg__(self):
        return self.from_torch(torch.neg([self]))

    def __not__(self):
        return self.from_torch(torch.logical_not([self]))

    def __getitem__(self, indices):
        res = self.value.__getitem__(indices)

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
        oname = _ox.slice(*self.my_args(), starts=bs, ends=es, axes=ds, steps=ss)
        if squeeze:  # single index means we must drop the axis
            oname = _ox.squeeze(*self.ox_name_args(oname), axes=squeeze)

        return self.from_torch(res, name=oname)

    @classmethod
    def ox_name_args(cls, input_name, output_n=1):
        container = cls.get_trace_session().container
        output_name = [_ox.get_unique_tensor_name(str(n_)) for n_ in range(output_n)]
        operator_name = None
        return input_name, output_name, container, operator_name

    def ox_args(cls, tensors, output_n=1):
        input_name = [ts_.name for ts_ in tensors]
        return cls.ox_name_args(input_name, output_n)

    def my_args(self):
        return self.ox_args([self])


    # def __getattribute__(self, attr):
    #     """
    #     A little hack that allows to call unary operators in a chaining fashion,
    #     e.g. x.shape() instead of ox.shape(x).
    #     """
    #     if attr in Tensor._all_ops:
    #         f = self.ox.__getattribute__(attr)
    #
    #         def call_it(*args, **kwargs):
    #             assert len(args) == 0, "In chaining expressions, only keyword args are allowed"
    #             assert "inputs" not in kwargs, "Chaining expressions do not currently support additional inputs"
    #             return f(self, *args, **kwargs)
    #
    #         return call_it
    #     else:
    #         return object.__getattribute__(self, attr)


    @staticmethod
    def normalize_seq(list_or_tuple):
        return [x.value.item() if isinstance(x, Tensor) else x for x in list_or_tuple]

    @property
    def value(self) -> torch.Tensor:
        return self._t

    def long(self):
        return self.from_torch(self._t.long())

    def cumsum(self, dim: _int, *, dtype: Optional[_dtype]=None):
        return self.from_torch(self._t.cumsum(dim, dtype=dtype))

    def size(self):
        return self.from_torch(self._t.size())

    def type(self, ty):
        return self.from_torch(self._t.type(ty))

    def to(self, device):
        return self.from_torch(self._t.to(device))




def empty(*size: _int, memory_format: Optional[memory_format]=None, out: Optional[Tensor]=None, dtype: _dtype=None, layout: _layout=strided, device: Union[_device, str, None]=None, requires_grad:_bool=False):  # noqa
    n_size = Tensor.normalize_seq(size)
    return Tensor.from_torch(torch.empty(*n_size, memory_format=memory_format, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad))


def zeros(*size: _int, out: Optional[Tensor]=None, dtype: _dtype=None, layout: _layout=strided, device: Union[_device, str, None]=None, requires_grad:_bool=False):  # noqa
    n_size = Tensor.normalize_seq(size)
    return Tensor.from_torch(torch.zeros(*n_size, out, dtype, layout, device, requires_grad))


def argmax(input: Tensor, dim: Optional[_int] = None, keepdim: _bool = False):
    return Tensor.from_torch(torch.zeros(input, dim, keepdim))


def cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim, *, out: Optional[Tensor] = None):
    res = torch.cat(tensors, dim, out=out)
    oname = _ox.concat(*Tensor.ox_args(tensors), dim)
    return Tensor.from_torch(res, oname)


class _TracingEagerOp(EagerOp):
    def __call__(self, *args, **kwargs):
        outseq = super().__call__(*args, **kwargs)
        # outputs = dict(zip([n_.name for n_ in self.ort_session.get_outputs()], outseq))

        outputs = [Tensor.from_onnx(self.ort_session, outseq[n_], out_.name
                                    ) for n_, out_ in enumerate(self.ort_session.get_outputs())]
        # FIXME: The tokenizer support attention_mask output
        outputs.append(Tensor([1] * outputs[0].value.shape[1], 'attention_mask'))

        return tuple(outputs)


def op_from_customop(op_type, *args, **kwargs):
    return _TracingEagerOp.from_customop(op_type, *args, **kwargs)


def op_from_model(path_or_model, *args, **kwargs):
    return _TracingEagerOp.from_model(path_or_model, *args, **kwargs)
