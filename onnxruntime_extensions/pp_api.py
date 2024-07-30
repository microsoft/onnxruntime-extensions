# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

from . import _extensions_pydll as _C
if not hasattr(_C, "create_processor"):
    raise ImportError("onnxruntime_extensions is not built with pre-processing API")

create_processor = _C.create_processor
load_images = _C.load_images
image_pre_process = _C.image_pre_process
tensor_result_get_at = _C.tensor_result_get_at

create_tokenizer = _C.create_tokenizer
batch_tokenize = _C.batch_tokenize
batch_detoeknize = _C.batch_detoeknize

delete_object = _C.delete_object
