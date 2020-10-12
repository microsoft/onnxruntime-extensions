# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
The entry point to onnxruntime custom op library
"""

__version__ = "0.0.1"
__author__ = "Microsoft"

from onnxconverter_common.onnx_fx import GraphFunctionType as Types

from ._ocos import get_library_path
from ._ocos import Opdef

onnx_op = Opdef.declare
