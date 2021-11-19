import torch
import builtins
import functools
import numpy as np
from onnx import onnx_pb as onnx_proto
from typing import List, Tuple, Optional, Union, Any, ContextManager, overload, Iterator, NamedTuple
from torch.types import _int, _float, _bool, Number, _dtype, _device, _qscheme, _size, _layout  # noqa
from torch import strided, memory_format, contiguous_format, StringType  # noqa

from ._onnx_ops import ox as _ox
from .._ortapi2 import OrtPyFunction


class _EagerTensor:
    def __init__(self, _t, name=None, sess=None, raw_data: Any = None):
        self._t = _t if isinstance(_t, torch.Tensor) else torch.tensor(_t)
        if isinstance(name, (tuple, list)):
            assert len(name) == 1, "Multiple names for one tensor!"
            name = name[0]
        self.name = '' if name is None else name
        self.raw_data = raw_data
        self.symbolic_shape = []

    def __repr__(self):
        if self.raw_data is not None:
            return "name: {}, \"{}\"".format(self.name, str(self.raw_data))
        else:
            return "name: {}, {}, dtype={}".format(self.name, repr(self._t), str(self._t.dtype))

    _all_ops = {}

    @property
    def value(self) -> Union[torch.Tensor, Any]:
        return self.raw_data if self.raw_data else self._t

    @property
    def t(self):
        return self._t

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def onnx_type(self):
        return self.to_onnx_type(self._t.dtype)

    @classmethod
    def is_numeric(cls, np_arr):
        return np_arr.dtype.kind in set('buifc')

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
    def get_trace_session(cls):
        if not hasattr(cls, '_active_session'):
            raise RuntimeError("the tracing not started yet!")
        return cls._active_session  # noqa

    @classmethod
    def get_container(cls):
        return cls.get_trace_session().container

    @classmethod
    def from_onnx(cls, raw_val, ort_sess, name):
        raw_data = None
        if cls.is_numeric(raw_val):
            val = torch.from_numpy(raw_val)
        else:
            # only keep the shape and the value was stored by it-self.
            val = torch.empty(*raw_val.shape, dtype=torch.uint8)
            raw_data = raw_val
        t = cls(val, name, ort_sess, raw_data)
        return t

    @classmethod
    def from_torch(cls, _t, name):
        t_name = name if name is not None else "id_{}".format(id(_t))
        ts = cls(_t, t_name)
        return ts

    @classmethod
    # torch.tensor prototype
    def mytensor(cls, data: Any, dtype: Optional[_dtype] = None, device: Union[_device, str, None] = None, requires_grad: _bool = False):  # noqa
        y = torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        val = _ox.make_tensor(cls.to_onnx_type(y.dtype), list(y.size()),
                              [data] if isinstance(data, (int, float, str, bool)) else data)
        s = _ox.constant([], [_ox.get_unique_tensor_name('const')], cls.get_container(), None, value=val)
        return cls.from_torch(y, s)

    def numpy(self):
        return self._t.numpy() if self.raw_data is None else self.raw_data

    def item(self):
        return self.numpy().item()

    def get_shape(self):
        return self.t.size() if len(self.symbolic_shape) == 0 else self.symbolic_shape

    def _to_binary_tensor_args(self, other):
        # convert self, other to [self, other], but if either is a number, convert that to a constant
        x, y = self, other
        if isinstance(y, (int, float, bool, np.ndarray)):
            y = self.mytensor(y)
        elif isinstance(x, (int, float, bool, np.ndarray)):
            x = self.mytensor(x)
        return x, y

    _dup_id = 0

    def __copy__(self):
        new_t = _EagerTensor.from_torch(self.t, self.name + '_{}'.format(_EagerTensor._dup_id))
        self._dup_id += 1
        new_t.raw_data = self.raw_data
        return new_t

    def __add__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.add(x0._t, x1._t)
        s = _ox.add(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __sub__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.sub(x0._t, x1._t)
        s = _ox.sub(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __mul__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.mul(x0._t, x1._t)
        s = _ox.mul(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __div__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.div(x0._t, x1._t)
        s = _ox.div(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __pow__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.pow(x0._t, x1._t)
        s = _ox.pow(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __matmul__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.matmul(x0._t, x1._t)
        s = _ox.matmul(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __lt__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.less(x0._t, x1._t)
        s = _ox.less(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __le__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.less_equal(x0._t, x1._t)
        s = _ox.less_equal(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __eq__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.equal(x0._t, x1._t)
        s = _ox.equal(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __ne__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.not_equal(x0._t, x1._t)
        s = _ox.not_equal(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __gt__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.greater(x0._t, x1._t)
        s = _ox.greater(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __ge__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.greater_equal(x0._t, x1._t)
        s = _ox.greater_equal(*_EagerTensor.ox_args([x0, x1]))
        return self.from_torch(y, s)

    def __invert__(self):
        if self.t.dtype is torch.bool:
            y = torch.logical_not(self.t)
            s = _ox.not_op(*self.my_args())
            return self.from_torch(y, s)
        else:
            raise NotImplementedError("no numeric tensor inverse supported yet.")

    def __neg__(self):
        y = torch.neg([self.t])
        s = _ox.neg(*self.my_args())
        return self.from_torch(y, s)

    def __not__(self):
        y = torch.logical_not(self.t)
        s = _ox.not_op(*self.my_args())
        return self.from_torch(y, s)

    def __or__(self, other):
        x0, x1 = self._to_binary_tensor_args(other)
        y = torch.logical_or(x0._t, x1._t)
        s = _ox.or_op(*_EagerTensor.ox_args([x0, x1]))
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

    def __getattribute__(self, attr):
        """
        A little hack that allows to call unary operators in a chaining fashion,
        e.g. x.shape() instead of ox.shape(x).
        """
        if attr in _EagerTensor._all_ops:
            f = _EagerTensor._all_ops[attr]
            return functools.partial(f, self)
        else:
            return object.__getattribute__(self, attr)

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
    def ort_verify(cls, ts_from, ts_to):
        result, model = cls.get_trace_session().runops(ts_from, ts_to)
        for idx in range(len(ts_to)):
            if not np.allclose(ts_to[idx].numpy(), result[idx]):
                # ONNX cannot be import globally, which is conflict with torch.onnx
                import onnx  # noqa
                onnx.save_model(model, 'mt_debmodel.onnx')
                raise RuntimeError("ONNXRuntime Result is not same pytorch!")

    def create_and_verify(self, value, name, additional_inputs=None):
        ts_y = self.from_torch(value, name)
        inputs = [self] + ([] if additional_inputs is None else additional_inputs)
        self.ort_verify(inputs, [ts_y])
        return ts_y

    @classmethod
    def ox_args(cls, tensors, output_names=None):
        input_names = [ts_ if isinstance(ts_, str) else ts_.name for ts_ in tensors]
        return cls.ox_name_args(input_names, output_names)

    def my_args(self):
        return self.ox_args([self])

    @staticmethod
    def normalize_seq(list_or_tuple):
        return [x.value.item() if isinstance(x, _EagerTensor) else x for x in list_or_tuple]

    @staticmethod
    def to_onnx_type(torch_type):
        ty_dict = {torch.bool: onnx_proto.TensorProto.BOOL,
                   torch.float32: onnx_proto.TensorProto.FLOAT,
                   torch.long: onnx_proto.TensorProto.INT64,
                   torch.int32: onnx_proto.TensorProto.INT32}
        # ...
        return ty_dict.get(torch_type, onnx_proto.TensorProto.STRING)

    def long(self):
        y = self._t.long()
        s = _ox.cast(*self.my_args(), to=onnx_proto.TensorProto.INT64)
        return self.create_and_verify(y, s[0])

    def cumsum(self, dim: _int, *, dtype: Optional[_dtype] = None):  # noqa
        y = self._t.cumsum(dim, dtype=dtype)
        s = _ox.cumsum(*self.my_args(), axis=dim)
        return self.create_and_verify(y, s[0])

    def size(self):
        y = self._t.size()
        s = _ox.shape(*self.my_args())
        return self.create_and_verify(y, s[0])

    def type(self, dtype: Union[str, _dtype], non_blocking: _bool=False):
        y = self._t.type(dtype, non_blocking)
        s = _ox.cast(*self.my_args(), to=self.to_onnx_type(dtype))
        return self.create_and_verify(y, s)

    def to(self, device):
        y = self._t.to(device)
        s = _ox.identity(*self.my_args())
        return self.create_and_verify(y, s[0])

    def cpu(self):
        y = self._t.cpu()
        s = _ox.identity(*self.my_args())
        return self.create_and_verify(y, s[0])

    def detach(self):
        y = self._t.detach()
        s = _ox.identity(*self.my_args())
        return self.create_and_verify(y, s[0])

    def clone(self):
        y = self._t.clone()
        s = _ox.identity(*self.my_args())
        return self.create_and_verify(y, s[0])

    def masked_fill(self, mask, value):
        y = self._t.masked_fill(mask.value, value)
        if not isinstance(value, _EagerTensor):
            value = _EagerTensor.mytensor(value)
        s = _ox.where(*_EagerTensor.ox_args([mask, value, self]))
        return self.create_and_verify(y, s[0], additional_inputs=[mask, value])

    def unsqueeze(self, dim: _int):
        y = self._t.unsqueeze(dim)
        s = _ox.unsqueeze(*self.my_args(), [dim])
        return self.create_and_verify(y, s[0])

    def squeeze(self, dim: _int):
        y = self._t.squeeze(dim)
        s = _ox.squeeze(*self.my_args(), [dim])
        return self.create_and_verify(y, s[0])


def _create_ox_sequence(*size):
    container = _EagerTensor.get_container()
    con_x = []
    if builtins.any(isinstance(n_, _EagerTensor) for n_ in size):
        for x in size:
            if isinstance(x, _EagerTensor):
                x_h = _ox.unsqueeze(*_EagerTensor.ox_args([x]))[0]
            else:
                x_c = _ox.make_tensor(onnx_proto.TensorProto.INT64, [1], [x])
                x_h = _ox.constant([], [_ox.get_unique_tensor_name('const')], container, None, value=x_c)[0]
            con_x.append(x_h)
        return _ox.concat(con_x, [_ox.get_unique_tensor_name('concat')], container, None)
    else:
        ts_size = _ox.make_tensor(onnx_proto.TensorProto.INT64, [len(size)], size)
        return _ox.constant([], [_ox.get_unique_tensor_name('const')], container, None, value=ts_size)


def _create_ox_sequence_constant(*size, init_value=None, onnx_type=None):
    if onnx_type is None:
        onnx_type = onnx_proto.TensorProto.FLOAT
    names = _create_ox_sequence(*size)
    ts_val = _ox.make_tensor(onnx_type, [1], [init_value])

    container = _EagerTensor.get_container()
    s = _ox.constant_of_shape(names, [_ox.get_unique_tensor_name('cos')], container, None, value=ts_val)
    return s[0]


def empty(*size: Union[_int, _EagerTensor], memory_format: Optional[memory_format] = None, out: Optional[_EagerTensor] = None,
          dtype: _dtype = None, layout: _layout = strided, device: Union[_device, str, None] = None,
          requires_grad: _bool = False) -> _EagerTensor:  # noqa

    if len(size) == 1 and isinstance(size[0], list):
        size = size[0]
    n_size = _EagerTensor.normalize_seq(size)
    y = torch.empty(*n_size, memory_format=memory_format, out=out,
                    dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    s = _create_ox_sequence_constant(*size, init_value=0., onnx_type=_EagerTensor.to_onnx_type(y.dtype))
    return _EagerTensor.from_torch(y, s)


def zeros(*size: Union[_int, _EagerTensor], out: Optional[_EagerTensor] = None, dtype: _dtype = None, layout: _layout = strided,
          device: Union[_device, str, None] = None, requires_grad: _bool = False) -> _EagerTensor:  # noqa

    if len(size) == 1 and isinstance(size[0], list):
        size = size[0]
    n_size = _EagerTensor.normalize_seq(size)
    y = torch.zeros(*n_size, out=out, dtype=dtype,
                    layout=layout, device=device, requires_grad=requires_grad)
    s = _create_ox_sequence_constant(*size, init_value=0, onnx_type=_EagerTensor.to_onnx_type(y.dtype))
    return _EagerTensor.from_torch(y, s)


def ones(*size: Union[_int, _EagerTensor], out: Optional[_EagerTensor] = None, dtype: _dtype = None, layout: _layout = strided,
          device: Union[_device, str, None] = None, requires_grad: _bool = False) -> _EagerTensor:  # noqa

    if len(size) == 1 and isinstance(size[0], list):
        size = size[0]
    n_size = _EagerTensor.normalize_seq(size)
    y = torch.ones(*n_size, out=out, dtype=dtype,
                   layout=layout, device=device, requires_grad=requires_grad)
    s = _create_ox_sequence_constant(*size, init_value=1, onnx_type=_EagerTensor.to_onnx_type(y.dtype))
    return _EagerTensor.from_torch(y, s)


def repeat(input_ts: _EagerTensor, *repeats: Union[_int, _EagerTensor]) -> _EagerTensor:  # noqa

    if len(repeats) == 1 and isinstance(repeats[0], list):
        repeats = repeats[0]
    n_size = _EagerTensor.normalize_seq(repeats)
    y = input_ts.t.repeat(*n_size)
    seq = _create_ox_sequence(*repeats)
    s = _ox.tile(*input_ts.my_args(), repeats=seq[0])
    return _EagerTensor.from_torch(y, s[0])


def argmax(input_ts: _EagerTensor, dim: Optional[_int] = None, keepdim: _bool = False) -> _EagerTensor:  # noqa
    y = torch.argmax(input_ts.value, dim, keepdim)
    s = _ox.argmax(*input_ts.my_args(), axis=dim, keepdims=keepdim)
    return _EagerTensor.from_torch(y, s)


def softmax(input_ts: _EagerTensor, dim: _int, dtype: Optional[_dtype]=None) -> _EagerTensor:
    y = torch.softmax(input_ts.value, dim, dtype)
    s = _ox.softmax(*input_ts.my_args(), axis=dim)
    return _EagerTensor.from_torch(y, s)


def cat(tensors: Union[Tuple[_EagerTensor, ...], List[_EagerTensor]],
        dim, *, out: Optional[_EagerTensor] = None) -> _EagerTensor:  # noqa
    res = torch.cat([t_.value for t_ in tensors], dim, out=out)
    oname = _ox.concat(*_EagerTensor.ox_args(tensors), dim)
    y = _EagerTensor.from_torch(res, oname[0])
    _EagerTensor.ort_verify(tensors, [y])
    return y


def all(input_ts: _EagerTensor, out: Optional[_EagerTensor]=None) -> _EagerTensor:  # noqa
    container = _EagerTensor.get_container()
    y = torch.all(input_ts.value)
    s_casted = _ox.cast(*input_ts.my_args(), to=onnx_proto.TensorProto.INT64)
    s_redm = _ox.reducemin(s_casted, [_ox.get_unique_tensor_name('reducemin')], container, None, axes=[-1])
    s0 = _ox.constant([], [_ox.get_unique_tensor_name('const')],
                      container, None, value=_ox.make_tensor(onnx_proto.TensorProto.INT64, [1], [0]))
    s = _ox.greater(s_redm + s0, [_ox.get_unique_tensor_name('greater')], container, None)
    return input_ts.create_and_verify(y, s[0])


def any(input_ts: _EagerTensor, out: Optional[_EagerTensor]=None) -> _EagerTensor:  # noqa
    container = _EagerTensor.get_container()
    y = torch.any(input_ts.value)
    s_casted = _ox.cast(*input_ts.my_args(), to=onnx_proto.TensorProto.INT64)
    s_redm = _ox.reducesum(s_casted, [_ox.get_unique_tensor_name('reducesum')], container, None, axes=[-1])
    s0 = _ox.constant([], [_ox.get_unique_tensor_name('const')],
                      container, None, value=_ox.make_tensor(onnx_proto.TensorProto.INT64, [1], [0]))
    s = _ox.greater(s_redm + s0, [_ox.get_unique_tensor_name('greater')], container, None)
    return input_ts.create_and_verify(y, s[0])


def reshape(input_ts: _EagerTensor, shape: _size):
    y = input_ts.t.reshape(shape)
    s = _ox.reshape(*input_ts.my_args(), desired_shape=shape)
    return input_ts.create_and_verify(y, s[0])


def transpose(input_ts: _EagerTensor, dim0: _int, dim1: _int):
    y = input_ts.t.transpose(dim0, dim1)
    axes = list(range(y.dim()))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    s = _ox.transpose(*input_ts.my_args(), perm=axes)
    return input_ts.create_and_verify(y, s[0])


class _LoopIterator:
    def __init__(self, ctx):
        self.context = ctx

    def __iter__(self):
        return self

    def __next__(self):
        if self.context.is_stopped():
            _EagerTensor.get_trace_session().pop_container()
            raise StopIteration
        return self.context.current()


class _ControlFlowContext:
    def __init__(self):
        self.condition_i = None
        self.condition = None
        self.loop_count = None
        self.iteration_num = None
        self.states_i = []
        self.loop_states = []
        self.scan_outputs = []
        self.sub_graph = None

    def flow_output(self, cond, *outputs):
        assert len(outputs) >= len(self.loop_states), "The loop body doesn't return enough objects"
        if self.sub_graph is None:
            trc = _EagerTensor.get_trace_session()
            self.sub_graph = trc.build_graph(trc.container,
                                             [self.iteration_num, self.condition] + self.loop_states,
                                             [cond] + list(outputs))

        self.condition = cond
        c_state = len(self.loop_states)
        self.loop_states = list(outputs[:c_state])
        if len(self.scan_outputs) == 0:
            sc = [_EagerTensor(torch.unsqueeze(sci_.value, 0), 'sc_' + sci_.name) for sci_ in outputs[c_state:]]
            self.scan_outputs = sc
        else:
            next_extra_vars = []
            for idx_, ext_ in enumerate(outputs[c_state:]):
                et = self.scan_outputs[idx_]
                next_extra_vars.append(_EagerTensor(
                    torch.cat([et.value, torch.unsqueeze(outputs[c_state + idx_].value, 0)]), name=et.name))
            self.scan_outputs = next_extra_vars
        self.iteration_num.value.add_(1)

    def current(self):
        return [self.iteration_num] + list(self.loop_states)

    def finalize(self):
        # generate the outputs from the enclosing scope variables
        full_outputs = [_EagerTensor(o_.value, 'lp_' + o_.name) for o_ in self.loop_states + self.scan_outputs]
        _ox.loop(*_EagerTensor.ox_args(
            [self.loop_count, self.condition_i] + list(self.states_i),
            [ts_.name for ts_ in full_outputs]), body=self.sub_graph)
        return tuple(full_outputs)

    def is_stopped(self):
        return self.condition.item() is False or self.iteration_num.item() >= self.loop_count.item()

    def loop(self, loop_c, condition, *states):
        self.condition = condition
        self.condition_i = condition
        self.states_i = states
        _EagerTensor.get_trace_session().stack_container()
        self.iteration_num = _EagerTensor.mytensor(0)
        # clone the variables for the sub graph.
        self.loop_states = [_EagerTensor(st_.value, st_.name) for st_ in states]
        self.loop_count = loop_c
        loop_b = _LoopIterator(self)
        return iter(loop_b)


def control_flow():
    return _ControlFlowContext()


class _TracingEagerOp(OrtPyFunction):
    def __call__(self, *args, **kwargs):
        np_args = [ts_.numpy() if isinstance(ts_, _EagerTensor) else ts_ for ts_ in args]
        outseq = super().__call__(*np_args, **kwargs)
        outseq = outseq if isinstance(outseq, (list, tuple)) else [outseq]

        outputs = [_EagerTensor.from_onnx(outseq[n_], self.ort_session, out_.name)
                   for n_, out_ in enumerate(self.ort_session.get_outputs())]

        y_names = [y.name for y in outputs]
        _ox.model_call(*_EagerTensor.ox_args(args, output_names=y_names), oxml=self.onnx_model)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]


def op_from_customop(op_type, *args, **kwargs) -> _TracingEagerOp:
    return _TracingEagerOp.from_customop(op_type, *args, **kwargs)


def op_from_model(path_or_model, *args, **kwargs) -> _TracingEagerOp:
    return _TracingEagerOp.from_model(path_or_model, *args, **kwargs)


_EagerTensor._all_ops = {'argmax': argmax,
                         'softmax': softmax,
                         'reshape': reshape,
                         'transpose': transpose,
                         'repeat': repeat,
                         'any': any,
                         'all': all}

tensor = _EagerTensor.mytensor
tensor_from_onnx = _EagerTensor.from_onnx
tensor_from_torch = _EagerTensor.from_torch
tensor_set_session = _EagerTensor.set_active_session
