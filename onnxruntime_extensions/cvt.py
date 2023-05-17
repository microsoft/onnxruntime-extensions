import json
from typing import Union

from ._cuops import CustomOpConverter


_is_torch_available = False
try:
    import torch
    _is_torch_available = True
    from ._torch_cvt import WhisperConverter
except ImportError:
    import warnings
    warnings.warn("The Whisper processor needs torch.onnx support, please install it")
    WhisperConverter = None


class HFTokenizerConverter(CustomOpConverter):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def bpe_tokenizer(self, **kwargs):
        hf_gpt2_tokenizer = self.tokenizer
        attrs = {'vocab': json.dumps(
            hf_gpt2_tokenizer.encoder, separators=(',', ':'))}
        sorted_merges = {v_: k_ for k_,
                         v_ in hf_gpt2_tokenizer.bpe_ranks.items()}
        attrs['merges'] = '\n'.join("{} {}".format(
            *sorted_merges[n_]) for n_ in range(len(sorted_merges)))
        attrs.update(**kwargs)
        return attrs

    def bpe_decoder(self, **kwargs):
        decoder = self.tokenizer.decoder
        id_vocab = "\n".join([decoder[_idx] for _idx in sorted(decoder)])
        # with open("id_vocab.txt", "w", encoding="utf-8") as f:
        #     f.write(id_vocab)
        byte_decoder = self.tokenizer.byte_decoder
        str_byte_decoder = "\n".join(["{}\t{}".format(
            ord(_c), str(byte_decoder[_c])) for _c in byte_decoder])
        # with open("byte_decoder.txt", "w", encoding="utf-8") as f:
        #     f.write(str_byte_decoder)
        all_special_ids = self.tokenizer.all_special_ids
        added_tokens = self.tokenizer.added_tokens_decoder
        str_all_special_ids = "\n".join([str(_id) for _id in all_special_ids])
        str_added_tokens = "\n".join(
            ["{}\t{}".format(str(_id), added_tokens[_id]) for _id in added_tokens])
        kwargs.update({
            "id_vocab": id_vocab,
            "byte_decoder": str_byte_decoder,
            "added_tokens": str_added_tokens,
            "all_special_ids": str_all_special_ids,
            "skip_special_tokens": kwargs.get("skip_special_tokens", False)
        })

        return kwargs

    def clip_tokenizer(self, **kwargs):
        hf_clip_tokenizer = self.tokenizer
        attrs = {'vocab': json.dumps(
            hf_clip_tokenizer.encoder, separators=(',', ':'))}
        sorted_merges = {v_: k_ for k_,
                         v_ in hf_clip_tokenizer.bpe_ranks.items()}
        attrs['merges'] = '\n'.join("{} {}".format(
            *sorted_merges[n_]) for n_ in range(len(sorted_merges)))
        attrs.update(**kwargs)
        return attrs

    def roberta_tokenizer(self, **kwargs):
        hf_roberta_tokenizer = self.tokenizer
        attrs = {'vocab': json.dumps(
            hf_roberta_tokenizer.encoder, separators=(',', ':'))}
        sorted_merges = {v_: k_ for k_,
                         v_ in hf_roberta_tokenizer.bpe_ranks.items()}
        attrs['merges'] = '\n'.join("{} {}".format(
            *sorted_merges[n_]) for n_ in range(len(sorted_merges)))
        attrs.update(**kwargs)
        return attrs


_PROCESSOR_DICT = {
    "ClipTokenizer": ('ClipTokenizer', HFTokenizerConverter.clip_tokenizer,
                      'BpeDecoder', HFTokenizerConverter.bpe_decoder),
    "ClipTokenizerFast": (HFTokenizerConverter.clip_tokenizer),
    "RobertaTokenizer": (HFTokenizerConverter.roberta_tokenizer),
}


def gen_processing_models(processor: Union[str, object],
                          pre_proc_only: bool=False,
                          post_proc_only: bool=False,
                          **kwargs):
    """
    Generate the pre- and post-processing ONNX model, basing on the name or HF class.

    Parameters
    ----------
    processor:
        the HF processor/tokenizer instance, or the name (str) of a Data Processor
    pre_proc_only: bool
        Only generating pre-processing model, skip the post-processing model
    post_proc_only: bool
        Only generating post-processing model, skip the pre-processing model
    kwargs:
        The arguments for generating models

    Returns
    -------
    ONNX-Models
        The pre- and post-processing ONNX models
    """
    if processor == "WhisperProcessor":
        _converter = WhisperConverter(**kwargs)
        return None if post_proc_only else _converter.pre_processing(),\
            None if pre_proc_only else _converter.post_processing()

    if processor in _PROCESSOR_DICT:
        pass
