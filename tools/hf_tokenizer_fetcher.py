# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This module provides functionality to fetch and convert Hugging Face tokenizers
for use with ONNX Runtime extensions.
"""
# formatted by ruff

import os
import shutil
import argparse
import tempfile
import numpy as np

# edit environment variables to avoid protobuf version mismatch
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers.convert_slow_tokenizer import SpmConverter  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from tokenizers import decoders, normalizers, pre_tokenizers, Regex  # noqa: E402


OrtxTokenizer = None
try:
    from onnxruntime_extensions.pp_api import Tokenizer as OrtxTokenizer
except ImportError:
    pass


def _get_prepend_scheme(add_prefix_space: bool, original_tokenizer) -> str:
    if add_prefix_space:
        prepend_scheme = "always"
        if not getattr(original_tokenizer, "legacy", True):
            prepend_scheme = "first"
    else:
        prepend_scheme = "never"
    return prepend_scheme


class Baichuan2Converter(SpmConverter):
    handle_byte_fallback = True

    def __init__(self, original_tokenizer):
        super().__init__(original_tokenizer)
        original_tokenizer.add_prefix_space = False

    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.convert_ids_to_tokens(0), 0.0),
            (self.original_tokenizer.convert_ids_to_tokens(1), 0.0),
            (self.original_tokenizer.convert_ids_to_tokens(2), 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        unk_id = 0
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def normalizer(self, proto):
        if getattr(self.original_tokenizer, "legacy", True):
            sequence = []
            if getattr(self.original_tokenizer, "add_prefix_space", True):
                sequence += [normalizers.Prepend(prepend="▁")]
            sequence += [normalizers.Replace(pattern=" ", content="▁")]
            return normalizers.Sequence(sequence)
        return None  # non-legacy, no normalizer

    def pre_tokenizer(self, replacement, add_prefix_space):
        if not getattr(self.original_tokenizer, "legacy", True):  # non-legacy, we need a replace
            prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
            return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme, split=False)
        else:
            return super().pre_tokenizer(replacement, add_prefix_space)


class ChatGlmConverter(SpmConverter):
    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        _normalizers = [
            normalizers.Strip(left=False, right=True),  # stripping is important
            normalizers.Replace(Regex(" {2,}"), "▁"),
        ]
        return normalizers.Sequence([normalizers.Precompiled(precompiled_charsmap)] + _normalizers)

    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = "always"
        if hasattr(self.original_tokenizer, "legacy") and not self.original_tokenizer.legacy:
            prepend_scheme = "first"
        return pre_tokenizers.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space, prepend_scheme=prepend_scheme
        )


JSON_TOKEN_CONVERTERS = {
    "BaichuanTokenizer": Baichuan2Converter,
    "ChatGLMTokenizer": ChatGlmConverter,
}


def convert_tokenizer(model_path, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if output_dir is None:
        if os.path.isdir(model_path):
            output_dir = model_path
        else:
            # create a temporary directory
            output_dir = tempfile.mkdtemp()
            tokenizer.save_pretrained(output_dir)
        json_path = os.path.join(output_dir, "tokenizer.json")

    if type(tokenizer).__name__ in JSON_TOKEN_CONVERTERS:
        GenericSpmConverter = JSON_TOKEN_CONVERTERS[type(tokenizer).__name__]

    converted = GenericSpmConverter(tokenizer).converted()
    converted.save(json_path)
    print(f"**Tokenizer saved to {json_path}")
    return output_dir


def validate_tokenizer(model_path, output_dir):
    test_sentence = "I like walking my cute dog\n and\x17 then, 生活的真谛是   \t\t\t\t \n\n61"
    if OrtxTokenizer is None:
        print("onnxruntime_extensions package was built with C API enabled, skipping tokenization test")
    ortx_tokenizer = OrtxTokenizer(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    expected_ids = tokenizer(test_sentence, return_tensors="np")["input_ids"]
    ortx_ids = np.asarray(ortx_tokenizer.tokenize(test_sentence))
    assert np.array_equal(expected_ids[0], ortx_ids), f"Tokenization mismatch: {expected_ids[0]} != {ortx_ids}"
    print("Tokenization test passed")


def download_tokenizer(tokenizer_dir, output_dir):
    try:
        from transformers.utils import cached_file

        resolved_full_file = cached_file(tokenizer_dir, "tokenizer.json")
        resolved_config_file = cached_file(tokenizer_dir, "tokenizer_config.json")
    except ImportError:
        raise ValueError(f"Directory '{tokenizer_dir}' not found and transformers is not available")
    if not os.path.exists(resolved_full_file):
        raise FileNotFoundError(f"Downloaded HF file '{resolved_full_file}' cannot be found")
    if os.path.dirname(resolved_full_file) != os.path.dirname(resolved_config_file):
        raise FileNotFoundError(
            f"Downloaded HF files '{resolved_full_file}' " f"and '{resolved_config_file}' are not in the same directory"
        )

    if output_dir is None:
        output_dir = os.path.dirname(resolved_full_file)
        print(f"Using {output_dir} as output directory")
    else:
        # copy the files to the output directory
        shutil.copy(resolved_full_file, output_dir)
        shutil.copy(resolved_config_file, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a tokenizer from Hugging Face model hub and convert it if necessary"
        "a command line example: hf_tokenizer_fetcher.py --model-path baichuan-inc/Baichuan2-7B-Chat"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path or name to tokenizer can be loaded by transformers.AutoTokenizer",
    )
    parser.add_argument(
        "--output-path", type=str, required=False, help="The directory to save the generated tokenizer files"
    )
    args = parser.parse_args()

    converted_tokenizer = {"Baichuan2", "chatglm"}
    need_convert = False
    for token in converted_tokenizer:
        if args.model_path.find(token) != -1:
            need_convert = True
            break

    if need_convert:
        output_dir = convert_tokenizer(args.model_path, args.output_path)
        validate_tokenizer(args.model_path, output_dir)
    else:
        download_tokenizer(args.model_path, args.output_path)
