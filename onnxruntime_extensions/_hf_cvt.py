# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
_hf_cvt.py: HuggingFace Tokenizer/Processor Converter
"""

import json
import onnx
from numpy import array as nparray
from functools import partial
from collections import namedtuple, OrderedDict

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

    def bert_tokenizer(self, **kwargs):
        hf_bert_tokenizer = self.tokenizer
        # has to be sorted since the id of token was generated automatically.
        ordered_vocab = OrderedDict(sorted(hf_bert_tokenizer.vocab.items(), key=lambda item: int(item[1])))
        vocab = '\n'.join(ordered_vocab.keys())
        attrs = dict(vocab=vocab)
        init_kwargs = hf_bert_tokenizer.init_kwargs
        attrs['do_lower_case'] = 1 if 'do_lower_case' in init_kwargs and init_kwargs.get('do_lower_case') else 0
        attrs['strip_accents'] = 1 if 'strip_accents' in init_kwargs and init_kwargs.get('strip_accents') else 0
        attrs.update(**kwargs)
        return attrs

    def bert_decoder(self, **kwargs):
        hf_bert_tokenizer = self.tokenizer
        attrs = {'vocab': json.dumps(
            hf_bert_tokenizer.ids_to_tokens, separators=(',', ':'))}
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

    def spm_tokenizer(self, **kwargs):
        attrs = {'model': read_file(self.tokenizer.vocab_file, 'rb')}
        attrs.update(**kwargs)
        return attrs

    def spm_decoder(self, **kwargs):
        attrs = {'model': read_file(self.tokenizer.vocab_file, 'rb')}
        attrs.update(**kwargs)
        return attrs


TokenOpParam = namedtuple("TokenOpParam",
                          ["pre_op", "pre_attribute_cvt",
                           "post_op", "post_attribute_cvt",
                           "default_inputs"],
                          defaults=(None, None, None, None, None))

# @formatter:off
_PROCESSOR_DICT = {
    "BertTokenizer":    TokenOpParam('BertTokenizer',   HFTokenizerConverter.bert_tokenizer,
                                     'BertDecoder',     HFTokenizerConverter.bpe_decoder, None),
    "DistilBertTokenizer":
                        TokenOpParam('BertTokenizer',   HFTokenizerConverter.bert_tokenizer,
                                     'BertDecoder',     HFTokenizerConverter.bpe_decoder, None),
    "GPT2Tokenizer":    TokenOpParam('Gpt2Tokenizer',   HFTokenizerConverter.bpe_tokenizer,
                                     'BpeDecoder',      HFTokenizerConverter.bpe_decoder, None),
    "ClipTokenizer":    TokenOpParam('ClipTokenizer',   HFTokenizerConverter.clip_tokenizer,
                                     'BpeDecoder',      HFTokenizerConverter.bpe_decoder, None),
    "RobertaTokenizer": TokenOpParam("RobertaTokenizer",    HFTokenizerConverter.roberta_tokenizer,
                                     None, None, None),
    "T5Tokenizer":      TokenOpParam("SentencepieceTokenizer",  HFTokenizerConverter.spm_tokenizer,
                                     "SentencepieceDecoder",    HFTokenizerConverter.spm_decoder,
                                     default_inputs={'add_eos': [True]}),
    "LlamaTokenizer":   TokenOpParam("SentencepieceTokenizer",  HFTokenizerConverter.spm_tokenizer,
                                     "SentencepieceDecoder",    HFTokenizerConverter.spm_decoder,
                                     default_inputs={'add_bos': [True]}),
    "FalconTokenizer":   TokenOpParam('ClipTokenizer',   HFTokenizerConverter.bpe_tokenizer,
                                      'BpeDecoder',      HFTokenizerConverter.bpe_decoder, None)
}
# @formatter:on


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
        with_default_inputs = kwargs.pop("WITH_DEFAULT_INPUTS", True)
        _cvt_op = self.cvt_quadruple.pre_op
        _cvt_func = self.cvt_quadruple.pre_attribute_cvt
        cvt = partial(_cvt_func, self.cvt_obj)
        g = SingleOpGraph.build_graph(_cvt_op, cvt=cvt, **kwargs)
        default_inputs = []
        if with_default_inputs:
            op_class = SingleOpGraph.get_op_class(_cvt_op)
            default_inputs = op_class.input_default_values()
            if default_inputs is None:
                return g

        # add default_inputs into initializers to simplify the model input
        n_inputs = len(default_inputs)
        if self.cvt_quadruple.default_inputs is not None:
            default_inputs.update(self.cvt_quadruple.default_inputs)
            if len(default_inputs) != n_inputs:
                raise ValueError("Op: {} does have the inputs from its TokenOpParam.".format(_cvt_op))

        new_initializers = []

        for k, v in default_inputs.items():
            input_value_info = next((i for i in g.input if i.name == k), None)
            if input_value_info is None:
                raise ValueError("The input {} is not found in the graph".format(k))

            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(input_value_info.type.tensor_type.elem_type)
            value = nparray(v, np_dtype)
            new_initializers.append(onnx.numpy_helper.from_array(value, k))
        g.initializer.extend(new_initializers)
        new_inputs = [i for i in g.input if i.name not in default_inputs]
        g.ClearField("input")
        g.input.extend(new_inputs)
        return g

    def post_processing(self, **kwargs):
        _cvt_op = self.cvt_quadruple.post_op
        _cvt_func = self.cvt_quadruple.post_attribute_cvt
        cvt = partial(_cvt_func, self.cvt_obj)
        return SingleOpGraph.build_graph(_cvt_op, cvt=cvt, **kwargs)
