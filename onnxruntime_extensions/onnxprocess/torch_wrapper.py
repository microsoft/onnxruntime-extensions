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
