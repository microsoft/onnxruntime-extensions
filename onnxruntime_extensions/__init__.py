# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
The entry point to onnxruntime custom op library
"""

__author__ = "Microsoft"

from ._version import __version__
from ._ocos import get_library_path  # noqa
from ._ocos import Opdef, PyCustomOpDef # noqa
from ._ocos import hash_64 # noqa
from ._ocos import enable_py_op  # noqa
from ._ocos import expand_onnx_inputs  # noqa
from ._ocos import hook_model_op  # noqa
from ._ocos import default_opset_domain  # noqa
from ._cuops import *  # noqa
from ._ortapi2 import OrtPyFunction as PyOrtFunction # backward compatibility
from ._ortapi2 import OrtPyFunction, optimize_model, make_onnx_model, ONNXRuntimeError


onnx_op = Opdef.declare
PyOp = PyCustomOpDef

