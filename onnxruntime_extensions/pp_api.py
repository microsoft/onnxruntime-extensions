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
create_tokenizer_with_options = _C.create_tokenizer_with_options
update_tokenizer_options = _C.update_tokenizer_options
batch_tokenize = _C.batch_tokenize
batch_detokenize = _C.batch_detokenize
_apply_chat_template = _C.apply_chat_template

delete_object = _C.delete_object


class Tokenizer:
    def __init__(self, tokenizer_dir, options = None):
        self.tokenizer = None
        self.options = {}

        if os.path.isdir(tokenizer_dir):
            if options is None:
                self.tokenizer = create_tokenizer(tokenizer_dir)
            else:
                # Convert values to lowercase strings if bool, otherwise str
                for k, v in options.items():
                    if isinstance(v, bool):
                        self.options[k] = str(v).lower()
                    else:
                        self.options[k] = str(v)
                self.tokenizer = create_tokenizer_with_options(tokenizer_dir, self.options)
        else:
            try:
                from transformers.utils import cached_file

                # Required files
                resolved_full_file = cached_file(tokenizer_dir, "tokenizer.json")
                resolved_config_file = cached_file(tokenizer_dir, "tokenizer_config.json")

                # Optional: attempt to download chat_template.jinja
                try:
                    cached_file(tokenizer_dir, "chat_template.jinja")
                except EnvironmentError:
                    # It is okay if this file does not exist as not every model has it
                    pass

                # Optional: chat_template.json (e.g., some models use this instead)
                try:
                    cached_file(tokenizer_dir, "chat_template.json")
                except EnvironmentError:
                    pass

            except ImportError:
                raise ValueError(
                    f"Directory '{tokenizer_dir}' not found and transformers is not available"
                )

            if not os.path.exists(resolved_full_file):
                raise FileNotFoundError(
                    f"Downloaded HF file '{resolved_full_file}' cannot be found"
                )

            if os.path.dirname(resolved_full_file) != os.path.dirname(resolved_config_file):
                raise FileNotFoundError(
                    f"Downloaded HF files '{resolved_full_file}' "
                    f"and '{resolved_config_file}' are not in the same directory"
                )

            tokenizer_dir = os.path.dirname(resolved_full_file)
            self.tokenizer = create_tokenizer(tokenizer_dir)

    def tokenize(self, text):
        if isinstance(text, (list, tuple)):
            return batch_tokenize(self.tokenizer, text)
        return batch_tokenize(self.tokenizer, [text])[0]
    
    def update_options(self, options: dict[str, str | int | bool]):
        # Update tokenizer options at runtime.
        for k, v in options.items():
            if isinstance(v, bool):
                self.options[k] = str(v).lower()
            else:
                self.options[k] = str(v)

        update_tokenizer_options(self.tokenizer, self.options)

    def detokenize(self, tokens):
        return batch_detokenize(self.tokenizer, [tokens])

    def apply_chat_template(self, chat, template="", tools="",add_generation_prompt=True, tokenize=False):
        result = _apply_chat_template(
            self.tokenizer, template, chat, tools, add_generation_prompt, tokenize)
        return tensor_result_get_at(result, 1 if tokenize else 0)

    def __del__(self):
        if delete_object and self.tokenizer:
            delete_object(self.tokenizer)
        self.tokenizer = None


class ImageProcessor:
    def __init__(self, processor_json):
        self.processor = create_processor(processor_json)

    def pre_process(self, images):
        if isinstance(images, str):
            images = [images]
        if isinstance(images, list):
            images = load_images(images)
        return image_pre_process(self.processor, images)

    @staticmethod
    def to_numpy(result, idx):
        return tensor_result_get_at(result, idx)

    def __del__(self):
        if delete_object and self.processor:
            delete_object(self.processor)
        self.processor = None
