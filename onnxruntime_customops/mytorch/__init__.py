"""
override the torch importing, to dump all torch operators during the processing code.
"""
try:
    import torch
except ImportError:
    raise RuntimeError("pytorch not installed, which is required by this ONNX build tool")

from ._tensor import *  # noqa

from .session import ONNXTraceSession


def start_trace(inputs):
    return ONNXTraceSession(inputs)
