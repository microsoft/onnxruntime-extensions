# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
_hf_cvt.py: HuggingFace Tokenizer/Processor Converter
"""

import json
from functools import partial

from ._cuops import CustomOpConverter, SingleOpGraph
from .util import read_file


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

    def t5_tokenizer(self, **kwargs):
        attrs = {'model': read_file(self.tokenizer.vocab_file, 'rb')}
        attrs.update(**kwargs)
        return attrs

    def t5_decoder(self, **kwargs):
        attrs = {'model': read_file(self.tokenizer.vocab_file, 'rb')}
        attrs.update(**kwargs)
        return attrs


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


class HFTokenizerOnnxGraph:
    @staticmethod
    def extract_cls_name(processor):
        cls_name = processor if isinstance(processor, str) else type(processor).__name__
        if cls_name.endswith("TokenizerFast"):
            cls_name = cls_name[:-len("Fast")]
        return cls_name

    @classmethod
    def is_supported(cls, processor):
        cls_name = cls.extract_cls_name(processor)
        return cls_name in _PROCESSOR_DICT

    def __init__(self, processor, **kwargs):
        cls_name = self.extract_cls_name(processor)
        self.cvt_quadruple = _PROCESSOR_DICT[cls_name]
        self.cvt_obj = HFTokenizerConverter(processor)

    def pre_processing(self, **kwargs):
        _cvt_op = self.cvt_quadruple[0]
        _cvt_func = self.cvt_quadruple[1]
        cvt = partial(_cvt_func, self.cvt_obj)
        return SingleOpGraph.build_graph(_cvt_op, cvt=cvt, **kwargs)

    def post_processing(self, **kwargs):
        _cvt_op = self.cvt_quadruple[2]
        _cvt_func = self.cvt_quadruple[3]
        cvt = partial(_cvt_func, self.cvt_obj)
        return SingleOpGraph.build_graph(_cvt_op, cvt=cvt, **kwargs)
