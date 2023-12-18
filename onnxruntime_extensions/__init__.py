# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
"""
The `onnxruntime-extensions` Python package offers an API that allows users to generate models for pre-processing and
post-processing tasks. In addition, it also provides an API to register custom operations implemented in Python.
This enables more flexibility and control over model execution, thus expanding the functionality of the ONNX Runtime.
"""

__author__ = "Microsoft"

__all__ = [
    'gen_processing_models',
    'ort_inference',
    'get_library_path',
    'Opdef', 'onnx_op', 'PyCustomOpDef', 'PyOp',
    'enable_py_op',
    'expand_onnx_inputs',
    'hook_model_op',
    'default_opset_domain',
    'OrtPyFunction', 'PyOrtFunction',
    'optimize_model',
    'make_onnx_model',
    'ONNXRuntimeError',
    'hash_64',
    '__version__',
]

from ._version import __version__
from ._ocos import get_library_path
from ._ocos import Opdef, PyCustomOpDef
from ._ocos import hash_64
from ._ocos import enable_py_op
from ._ocos import expand_onnx_inputs
from ._ocos import hook_model_op
from ._ocos import default_opset_domain
from ._cuops import *  # noqa
from ._ortapi2 import OrtPyFunction as PyOrtFunction  # backward compatibility
from ._ortapi2 import OrtPyFunction, ort_inference, optimize_model, make_onnx_model
from ._ortapi2 import ONNXRuntimeError, ONNXRuntimeException
from .cvt import gen_processing_models

# rename the implementation with a more formal name
onnx_op = Opdef.declare
PyOp = PyCustomOpDef
