from typing import List, Tuple, Optional, Union, Any, ContextManager, overload, Iterator, NamedTuple  # noqa
from torch import (dtype,
                   tensor as _tensor,
                   Tensor as _Tensor)  # noqa
from torch import (float32,
                   float,
                   float64,
                   double,
                   float16,
                   bfloat16,
                   half,
                   uint8,
                   int8,
                   int16,
                   short,
                   int32,
                   int,
                   int64,
                   long,
                   complex32,
                   complex64,
                   cfloat,
                   complex128,
                   cdouble,
                   quint8,
                   qint8,
                   qint32,
                   bool)  # noqa
from torch.types import _int, _float, _bool, Number, _dtype, _device, _qscheme, _size, _layout  # noqa
from torch import (argmax as _argmax,
                   cat as _cat,
                   empty as _empty,
                   zeros as _zeros,
                   ones as _ones,
                   all as _all,
                   from_numpy as _from_numpy,
                   empty as _empty)  # noqa


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
    def __init__(self, _t):
        self._t = _t


def argmax(input: Tensor, dim: Optional[_int]=None, keepdim: _bool=False):
    return _argmax(input, dim, keepdim)


def cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim, *, out: Optional[Tensor]=None):
    return _cat(tensors, dim, out=out)


def tensor(data: Any, dtype: Optional[_dtype]=None, device: Union[_device, str, None]=None, requires_grad: _bool=False):
    return _tensor(data, dtype, device, requires_grad)
