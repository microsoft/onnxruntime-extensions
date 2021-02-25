import onnx
import torch

from typing import List, Tuple, Optional, Union, Any, ContextManager, overload, Iterator, NamedTuple  # noqa
from torch.types import _int, _float, _bool, Number, _dtype, _device, _qscheme, _size, _layout  # noqa
from torch import strided, memory_format, contiguous_format  # noqa

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
    def from_torch(_t):
        ts = Tensor(_t, name="id_{}".format(id(_t)))
        return ts

    def __init__(self, _t, name=None):
        # FIXME: the default is int64?
        self._t = _t if isinstance(_t, torch.Tensor) else torch.tensor(_t, dtype=torch.int64)
        self.name = '' if name is None else name

    def __repr__(self):
        return "name: {}, {}".format(self.name, repr(self._t))

    @property
    def value(self):
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
    return Tensor.from_torch(torch.empty(size, memory_format, out, dtype, layout, device, requires_grad))


def zeros(*size: _int, out: Optional[Tensor]=None, dtype: _dtype=None, layout: _layout=strided, device: Union[_device, str, None]=None, requires_grad:_bool=False):  # noqa
    return Tensor.from_torch(torch.zeros(*size, out, dtype, layout, device, requires_grad))


def argmax(input: Tensor, dim: Optional[_int] = None, keepdim: _bool = False):
    return Tensor.from_torch(torch.zeros(input, dim, keepdim))


def cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim, *, out: Optional[Tensor] = None):
    return Tensor.from_torch(torch.cat(tensors, dim, out=out))


class EagerOpTs(EagerOp):
    def __call__(self, *args, **kwargs):
        outseq = super().__call__(*args, **kwargs)
        # outputs = dict(zip([n_.name for n_ in self.ort_session.get_outputs()], outseq))

        outputs = [Tensor.from_onnx(self.ort_session, outseq[n_], out_.name
                                    ) for n_, out_ in enumerate(self.ort_session.get_outputs())]
        # FIXME: The tokenizer support attention_mask output
        outputs.append(Tensor([1] * outputs[0].value.shape[1], 'attention_mask'))

        return tuple(outputs)


def op_from_customop(op_type, *args, **kwargs):
    return EagerOpTs.from_customop(op_type, *args, **kwargs)


def op_from_model(path_or_model, *args, **kwargs):
    return EagerOpTs.from_model(path_or_model, *args, **kwargs)
