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


_is_torch_available = False
try:
    import torch    # noqa
    _is_torch_available = True
    from ._torch_cvt import WhisperDataProcGraph
except ImportError:
    WhisperDataProcGraph = None


_PRE_POST_PAIR = {'TrieTokenizer': "TrieDetokenizer"}


def gen_processing_models(processor: Union[str, object],
                          pre_kwargs: dict = None,
                          post_kwargs: dict = None,
                          opset: int = None,
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
