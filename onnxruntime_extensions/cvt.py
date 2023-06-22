# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
cvt.py: Processing Graph Converter and Generator
"""

from typing import Union
from functools import partial

from ._cuops import SingleOpGraph
from ._hf_cvt import HFTokenizerConverter
from ._ortapi2 import make_onnx_model


_is_torch_available = False
try:
    import torch
    _is_torch_available = True
    from ._torch_cvt import WhisperConverter
except ImportError:
    import warnings
    warnings.warn("The Whisper processor needs torch.onnx support, please install it")
    WhisperConverter = None


_PROCESSOR_DICT = {
    "GPT2Tokenizer":    ('Gpt2Tokenizer', HFTokenizerConverter.bpe_tokenizer,
                         'BpeDecoder', HFTokenizerConverter.bpe_decoder), 
    "ClipTokenizer":    ('ClipTokenizer', HFTokenizerConverter.clip_tokenizer,
                         'BpeDecoder', HFTokenizerConverter.bpe_decoder),
    "RobertaTokenizer": ("RobertaTokenizer", HFTokenizerConverter.roberta_tokenizer,
                         None, None),
    "T5Tokenizer":      ("SentencepieceTokenizer", HFTokenizerConverter.t5_tokenizer,
                         "SentencepieceDecoder", HFTokenizerConverter.t5_decoder),
}


def gen_processing_models(processor: Union[str, object],
                          pre_kwargs: dict=None,
                          post_kwargs: dict=None,
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
    post_kwargs: dict
        Keyword arguments for generating the post-processing model
    kwargs:
        The additional arguments for generating models

    Returns
    -------
    ONNX-Models
        The pre- and post-processing ONNX models
    """
    pre_g = None
    post_g = None

    if pre_kwargs is None and post_kwargs is None:
        raise ValueError("Either pre_kwargs or post_kwargs should be provided. None means no processing")

    cls_name = processor if isinstance(processor, str) else type(processor).__name__
    if cls_name.endswith("TokenizerFast"):
        cls_name = cls_name[:-len("Fast")]

    if cls_name == "WhisperProcessor":
        _converter = WhisperConverter(**kwargs)
        pre_g = _converter.pre_processing(**pre_kwargs) if pre_kwargs is not None else None
        post_g = _converter.post_processing(**post_kwargs) if post_kwargs is not None else None

    if cls_name in _PROCESSOR_DICT:
        cvt_quadr = _PROCESSOR_DICT[cls_name]
        _cvt_op = cvt_quadr[0]
        _cvt_func = cvt_quadr[1]
        cvt_obj = HFTokenizerConverter(processor)
        cvt = partial(_cvt_func, cvt_obj)

        if pre_kwargs is not None:
            pre_g = SingleOpGraph.build_graph(_cvt_op, cvt=cvt, **pre_kwargs)
        if post_kwargs is not None:
            _cvt_op = cvt_quadr[2]
            _cvt_func = cvt_quadr[3]
            cvt = partial(_cvt_func, cvt_obj)
            post_g = SingleOpGraph.build_graph(_cvt_op, cvt=cvt, **post_kwargs)

    return make_onnx_model(pre_g) if pre_g else None, \
        make_onnx_model(post_g) if post_g else None
