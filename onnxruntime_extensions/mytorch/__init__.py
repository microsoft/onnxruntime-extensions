"""
override the torch importing, to dump all torch operators during the processing code.
!!!This package depends on onnxruntime_extensions root package, but not vice versa.!!!
, since this package fully relies on pytorch, while the onnxruntime_extensions doesn't
"""
try:
    import torch
except ImportError:
    raise RuntimeError("pytorch not installed, which is required by this ONNX build tool")

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

from torch import randn, onnx  # noqa


from ._tensor import *  # noqa
from ._builder import build_customop_model
from ._session import ONNXTraceSession

trace_for_onnx = ONNXTraceSession.trace_for_onnx
