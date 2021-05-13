"""
override the torch importing, to dump all torch operators during the processing code.
!!!This package depends on onnxruntime_extensions root package, but not vice versa.!!!
, since this package fully relies on pytorch, while the onnxruntime_extensions doesn't
"""

from . _tensor import op_from_customop as pyfunc_from_custom_op
from . _tensor import op_from_model as pyfunc_from_model
from ._builder import build_customop_model
from ._session import ONNXTraceSession

trace_for_onnx = ONNXTraceSession.trace_for_onnx
