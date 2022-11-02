import json
from collections import OrderedDict

from ._base import ProcessingTracedModule, tensor_data_type as _dt
from ._torchext import create_op_function
from ._onnx_ops import schema
from .._ocos import default_opset_domain


def make_custom_op(ctx, op_type, input_names, output_names, container, operator_name=None, **kwargs):
    op_name = container.get_unique_operator_name(op_type) if operator_name is None else operator_name
    container.add_node(op_type, input_names, output_names,
                       op_version=1, name=op_name, op_domain=default_opset_domain(), **kwargs)


def create_bert_tokenizer(ctx, name, input_names, output_names, container, operator_name=None, **kwargs):
    if 'hf_tok' in kwargs:
        hf_bert_tokenizer = kwargs['hf_tok']
        ordered_vocab = OrderedDict(sorted(hf_bert_tokenizer.vocab.items(), key=lambda item: int(item[1])))
        vocab = '\n'.join(ordered_vocab.keys())
        attrs = dict(vocab_file=vocab)
        # Unfortunately, there's no specific accessor function on
        # transformers.BertTokenizer to query for strip_accents.
        attrs['strip_accents'] = 1 if 'strip_accents' in hf_bert_tokenizer.init_kwargs and hf_bert_tokenizer.init_kwargs.get('strip_accents') else 0
        attrs['do_lower_case'] = 1 if hasattr(hf_bert_tokenizer, 'do_lower_case') and hf_bert_tokenizer.do_lower_case else 0
    elif 'vocab_file' in kwargs:
        vocab = None
        vocab_file = kwargs['vocab_file']
        with open(vocab_file, "r", encoding='utf-8') as vf:
            lines = vf.readlines()
            vocab = '\n'.join(lines)
        if vocab is None:
            raise RuntimeError("Cannot load vocabulary file {}!".format(vocab_file))
        attrs = dict(vocab_file=vocab)
        if 'strip_accents' in kwargs:
            attrs['strip_accents'] = kwargs['strip_accents']
        if 'do_lower_case' in kwargs:
            attrs['do_lower_case'] = kwargs['do_lower_case']
    else:
        raise RuntimeError("Need hf_tok/vocab_file parameter to build the tokenizer")

    return make_custom_op(ctx, name, input_names,
                          output_names, container, operator_name=operator_name, **attrs)


@schema(inputs=((_dt.STRING, []),),
        outputs=((_dt.INT64, []), (_dt.INT64, []), (_dt.INT64, [])))
def bert_tokenizer(ctx, input_names, output_names, container, operator_name=None, **kwargs):
    return create_bert_tokenizer(ctx, 'BertTokenizer', input_names, output_names,
                                 container, operator_name=operator_name, **kwargs)


@schema(inputs=((_dt.STRING, []),),
        outputs=((_dt.INT64, []), (_dt.INT64, []), (_dt.INT64, [])))
def hf_bert_tokenizer(ctx, input_names, output_names, container, operator_name=None, **kwargs):
    return create_bert_tokenizer(ctx, 'HfBertTokenizer', input_names, output_names,
                                 container, operator_name=operator_name, **kwargs)


@schema(inputs=((_dt.STRING, []),),
        outputs=((_dt.INT64, []), (_dt.INT64, [])))
def gpt2_tokenize(ctx, input_names, output_names, container, operator_name=None, **kwargs):
    if 'hf_tok' in kwargs:
        hf_gpt2_tokenizer = kwargs['hf_tok']
        attrs = {'vocab': json.dumps(hf_gpt2_tokenizer.encoder, separators=(',', ':'))}
        sorted_merges = {v_: k_ for k_, v_ in hf_gpt2_tokenizer.bpe_ranks.items()}
        attrs['merges'] = '\n'.join("{} {}".format(*sorted_merges[n_]) for n_ in range(len(sorted_merges)))
    elif 'vocab' in kwargs:
        attrs = dict(
            vocab=kwargs['vocab'],
            merges=kwargs['merges'])
    else:
        raise RuntimeError("Need hf_tok/vocab parameter to build the tokenizer")
    padding_len = -1
    if 'padding_length' in kwargs:
        padding_len = kwargs['padding_length']
    attrs['padding_length'] = padding_len

    return make_custom_op(ctx, 'GPT2Tokenizer', input_names,
                          output_names, container, operator_name=operator_name, **attrs)


def _get_file_content(path):
    with open(path, "rb") as file:
        return file.read()


def _get_bound_object(func):
    return func.__self__

# v1. Order of outputs - input_ids, token_type_ids, attention_mask
#    (this is NOT consistent with the HuggingFace implementation of the tokenizer)
class PreHuggingFaceBert(ProcessingTracedModule):
    def __init__(self, hf_tok=None, vocab_file=None, do_lower_case=0, strip_accents=1):
        super(PreHuggingFaceBert, self).__init__()
        if hf_tok is None:
            self.onnx_bert_tokenizer = create_op_function('BertTokenizer', bert_tokenizer,
                                                          vocab_file=vocab_file,
                                                          do_lower_case=do_lower_case,
                                                          strip_accents=strip_accents)
        else:
            self.onnx_bert_tokenizer = create_op_function('BertTokenizer', bert_tokenizer,
                                                          hf_tok=hf_tok)

    def forward(self, text):
        return self.onnx_bert_tokenizer(text)

    def export(self, *args, **kwargs):
        return _get_bound_object(self.onnx_bert_tokenizer).build_model(kwargs.get('opset_version', 0), *args)


# v2. Order of outputs - input_ids, attention_mask, token_type_ids
#    (this is consistent with the HuggingFace implementation of the tokenizer)
class HfBertTokenizer(ProcessingTracedModule):
    def __init__(self, hf_tok=None, vocab_file=None, do_lower_case=0, strip_accents=1):
        super(HfBertTokenizer, self).__init__()
        if hf_tok is None:
            self.onnx_bert_tokenizer = create_op_function('HfBertTokenizer', hf_bert_tokenizer,
                                                          vocab_file=vocab_file,
                                                          do_lower_case=do_lower_case,
                                                          strip_accents=strip_accents)
        else:
            self.onnx_bert_tokenizer = create_op_function('HfBertTokenizer', hf_bert_tokenizer,
                                                          hf_tok=hf_tok)

    def forward(self, text):
        return self.onnx_bert_tokenizer(text)

    def export(self, *args, **kwargs):
        return _get_bound_object(self.onnx_bert_tokenizer).build_model(kwargs.get('opset_version', 0), *args)


class PreHuggingFaceGPT2(ProcessingTracedModule):
    def __init__(self, hf_tok=None, vocab_file=None, merges_file=None, padding_length=-1):
        super(PreHuggingFaceGPT2, self).__init__()
        if hf_tok is None:
            self.onnx_gpt2_tokenize = create_op_function('GPT2Tokenizer', gpt2_tokenize,
                                                         vocab=_get_file_content(vocab_file),
                                                         merges=_get_file_content(merges_file),
                                                         padding_length=padding_length)
        else:
            self.onnx_gpt2_tokenize = create_op_function('GPT2Tokenizer', gpt2_tokenize, hf_tok=hf_tok)

    def forward(self, text):
        return self.onnx_gpt2_tokenize(text)

    def export(self, *args, **kwargs):
        return _get_bound_object(self.onnx_gpt2_tokenize).build_model(kwargs.get('opset_version', 0), *args)
