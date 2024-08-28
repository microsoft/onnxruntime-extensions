# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import os
from . import _extensions_pydll as _C
if not hasattr(_C, "delete_object"):
    raise ImportError(
        "onnxruntime_extensions is not built with pre-processing C API\n"
        "To enable it, please build the package with --ortx-user-option=pp_api")

create_processor = _C.create_processor
load_images = _C.load_images
image_pre_process = _C.image_pre_process
tensor_result_get_at = _C.tensor_result_get_at

create_tokenizer = _C.create_tokenizer
batch_tokenize = _C.batch_tokenize
batch_detokenize = _C.batch_detokenize

delete_object = _C.delete_object


class Tokenizer:
    def __init__(self, tokenizer_dir):
        self.tokenizer = None
        if os.path.isdir(tokenizer_dir):
            self.tokenizer = create_tokenizer(tokenizer_dir)
        else:
            try:
                from transformers.utils import cached_file
                resolved_full_file = cached_file(
                    tokenizer_dir, "tokenizer.json")
                resolved_config_file = cached_file(
                    tokenizer_dir, "tokenizer_config.json")
            except ImportError:
                raise ValueError(
                    f"Directory '{tokenizer_dir}' not found and transformers is not available")
            if not os.path.exists(resolved_full_file):
                raise FileNotFoundError(
                    f"Downloaded HF file '{resolved_full_file}' cannot be found")
            if (os.path.dirname(resolved_full_file) != os.path.dirname(resolved_config_file)):
                raise FileNotFoundError(
                    f"Downloaded HF files '{resolved_full_file}' "
                    f"and '{resolved_config_file}' are not in the same directory")

            tokenizer_dir = os.path.dirname(resolved_full_file)
            self.tokenizer = create_tokenizer(tokenizer_dir)

    def tokenize(self, text):
        return batch_tokenize(self.tokenizer, [text])[0]

    def detokenize(self, tokens):
        return batch_detokenize(self.tokenizer, [tokens])[0]

    def __del__(self):
        if delete_object and self.tokenizer:
            delete_object(self.tokenizer)
        self.tokenizer = None


class ImageProcessor:
    def __init__(self, processor_json):
        self.processor = create_processor(processor_json)

    def pre_process(self, images):
        return image_pre_process(self.processor, images)

    def __del__(self):
        if delete_object and self.processor:
            delete_object(self.processor)
        self.processor = None
