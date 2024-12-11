# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
cvt.py: Processing Graph Converter and Generator
"""

from typing import Union

from ._hf_cvt import HFTokenizerConverter, HFTokenizerOnnxGraph  # noqa
from ._ortapi2 import make_onnx_model, SingleOpGraph

import os
import numpy as np
import tempfile
import shutil

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

_is_torch_available = False
try:
    import torch    # noqa
    _is_torch_available = True
    from ._torch_cvt import WhisperDataProcGraph
except ImportError:
    WhisperDataProcGraph = None


_PRE_POST_PAIR = {'TrieTokenizer': "TrieDetokenizer"}

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

# Save tokenizer JSON files using HuggingFace AutoTokenizer
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

# Validate tokenizer files downloaded from cache
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

# Download tokenizer JSON files from cache
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

    if output_dir is None or len(output_dir) == 0:
        output_dir = os.path.dirname(resolved_full_file)
        print(f"Using {output_dir} as output directory")
        return output_dir
    else:
        # copy the files to the output directory
        shutil.copy(resolved_full_file, output_dir)
        shutil.copy(resolved_config_file, output_dir)
        return output_dir


def gen_processing_models(processor: Union[str, object],
                          pre_kwargs: dict = None,
                          post_kwargs: dict = None,
                          opset: int = None,
                          cached: bool = False,
                          **kwargs):
    """
    Generate the pre- and post-processing ONNX model, basing on the name or HF class.

    Parameters
    ----------
    processor:
        the HF processor/tokenizer instance, or the name (str) of a Data Processor
        the instance is preferred, otherwise when name was given, the corresponding configuration for the processor
        has to be provided in the kwargs
    pre_kwargs: dict
        Keyword arguments for generating the pre-processing model
        WITH_DEFAULT_INPUTS: bool, add default inputs to the graph, default is True
        CAST_TOKEN_ID: bool, add a cast op to output token IDs to be int64 if needed, default is False
    post_kwargs: dict
        Keyword arguments for generating the post-processing model
    opset: int
        the target opset version of the model
    cached: bool
        the flag for using cached tokenizer files; this option leverages the blob-loading functionality
        which loads HF tokenizers from memory rather than using the tokenizer files in HF JSON format.
    kwargs:
        The additional arguments for generating models

    Returns
    -------
    ONNX-Models
        The pre- and post-processing ONNX models
    """
    if pre_kwargs is None and post_kwargs is None:
        raise ValueError(
            "Either pre_kwargs or post_kwargs should be provided. None means no processing graph output.")

    # If true, we get the tokenizer JSON files by either downloading from cache or using HuggingFace AutoTokenizer
    # to convert them, and then create an ONNX model with the JSON files as strings in the model attributes (attrs).
    if cached:
        model_name = processor if isinstance(processor, str) else type(processor).__name__

        converted_tokenizer = {"Baichuan2", "chatglm"}
        need_convert = False
        for token in converted_tokenizer:
            if model_name.find(token) != -1:
                need_convert = True
                break

        if need_convert:
            model_dir = convert_tokenizer(model_name)
            validate_tokenizer(model_name, None)
        else:
            model_dir = download_tokenizer(model_name, None)

        # Load the content of tokenizer.json into a string
        with open(f"{model_dir}/tokenizer.json", "r", encoding="utf-8") as f:
            tokenizer_vocab = f.read()

        # Load the content of tokenizer_config.json into a string
        with open(f"{model_dir}/tokenizer_config.json", "r", encoding="utf-8") as f:
            tokenizer_config = f.read()

        # Create an ONNX model with these JSON file strings in attrs
        g_pre, g_post = (None, None)
        if pre_kwargs is not None:
            # Add tokenizer_vocab and tokenizer_config to the kwargs
            # so they are added to attrs in build_graph
            pre_kwargs['tokenizer_vocab'] = tokenizer_vocab
            pre_kwargs['tokenizer_config'] = tokenizer_config
            g_pre = SingleOpGraph.build_graph("HfJsonTokenizer", **pre_kwargs)
        if post_kwargs is not None:
            if pre_kwargs is None:
                cls_name = processor
            else:
                if processor not in _PRE_POST_PAIR:
                    raise RuntimeError(
                        f"Cannot locate the post processing operator name from {processor}")
                cls_name = _PRE_POST_PAIR[processor]
            # Add tokenizer_vocab and tokenizer_config to the kwargs
            # so they are added to attrs in build_graph
            post_kwargs['tokenizer_vocab'] = tokenizer_vocab
            post_kwargs['tokenizer_config'] = tokenizer_config
            g_post = SingleOpGraph.build_graph(cls_name, **post_kwargs)
        return make_onnx_model(g_pre) if g_pre else None, make_onnx_model(g_post) if g_post else None
    else:
        if isinstance(processor, str):
            g_pre, g_post = (None, None)
            if pre_kwargs:
                g_pre = SingleOpGraph.build_graph(processor, **pre_kwargs)
            if post_kwargs:
                if pre_kwargs is None:
                    cls_name = processor
                else:
                    if processor not in _PRE_POST_PAIR:
                        raise RuntimeError(
                            f"Cannot locate the post processing operator name from {processor}")
                    cls_name = _PRE_POST_PAIR[processor]
                g_post = SingleOpGraph.build_graph(cls_name, **post_kwargs)
            return make_onnx_model(g_pre) if g_pre else None, make_onnx_model(g_post) if g_post else None

        cls_name = type(processor).__name__
        if cls_name == "WhisperProcessor":
            if WhisperDataProcGraph is None:
                raise ValueError(
                    "The Whisper processor needs torch.onnx support, please install pytorch 2.0 and above")
            _converter = WhisperDataProcGraph(processor, opset=opset, **kwargs)
            pre_m = _converter.pre_processing(
                **pre_kwargs) if pre_kwargs is not None else None
            post_m = _converter.post_processing(
                **post_kwargs) if post_kwargs is not None else None
            return pre_m, post_m
        elif HFTokenizerOnnxGraph.is_supported(processor):
            _converter = HFTokenizerOnnxGraph(processor)
            pre_g = _converter.pre_processing(
                **pre_kwargs) if pre_kwargs is not None else None
            post_g = _converter.post_processing(
                **post_kwargs) if post_kwargs is not None else None
            return make_onnx_model(pre_g) if pre_g else None, \
                make_onnx_model(post_g) if post_g else None
        else:
            raise ValueError(f"Unsupported processor/tokenizer: {cls_name}")
