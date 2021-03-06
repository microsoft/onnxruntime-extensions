"""
override the torch importing, to dump all torch operators during the processing code.
!!!This package depends on onnxruntime_customops root package, but not vice versa.!!!
, since this package fully relies on pytorch, while the onnxruntime_customops doesn't
"""
try:
    import torch
except ImportError:
    raise RuntimeError("pytorch not installed, which is required by this ONNX build tool")

from ._tensor import *  # noqa
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
from ._session import ONNXTraceSession, build_customop_model


start_trace = ONNXTraceSession.start_trace
stop_trace = ONNXTraceSession.stop_trace
