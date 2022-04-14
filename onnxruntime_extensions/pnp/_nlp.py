import json

from ._base import ProcessingTracedModule, tensor_data_type as _dt
from ._torchext import create_op_function
from ._onnx_ops import schema
from .._ocos import default_opset_domain


def make_custom_op(ctx, op_type, input_names, output_names, container, operator_name=None, **kwargs):
    op_name = container.get_unique_operator_name(op_type) if operator_name is None else operator_name
    container.add_node(op_type, input_names, output_names,
                       op_version=1, name=op_name, op_domain=default_opset_domain(), **kwargs)


@schema(inputs=((_dt.STRING, []),),
        outputs=((_dt.INT64, []), (_dt.INT64, []), (_dt.INT64, [])))
def bert_tokenize(ctx, input_names, output_names, container, operator_name=None, **kwargs):
    if 'hf_tok' in kwargs:
        # TODO: need bert-tokenizer support JSON format
        hf_bert_tokenizer = kwargs['hf_tok']
        attrs = {'vocab_file': json.dumps(hf_bert_tokenizer.vocab, separators=(',', ':'))}
    elif 'vocab_file' in kwargs:
        attrs = dict(vocab_file=kwargs['vocab_file'])
    else:
        raise RuntimeError("Need hf_tok/vocab_file parameter to build the tokenizer")
    if 'strip_accents' in kwargs:
        strip_accents = kwargs['strip_accents']
        attrs['strip_accents'] = strip_accents

    return make_custom_op(ctx, 'BertTokenizer', input_names,
                          output_names, container, operator_name=operator_name, **attrs)


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


class PreHuggingFaceBert(ProcessingTracedModule):
    def __init__(self, hf_tok=None, vocab_file=None, do_lower_case=0, strip_accents=1):
        super(PreHuggingFaceBert, self).__init__()
        if hf_tok is None:
            self.onnx_bert_tokenize = create_op_function('BertTokenizer', bert_tokenize,
                                                         vocab_file=_get_file_content(vocab_file),
                                                         do_lower_case=do_lower_case,
                                                         strip_accents=strip_accents)
        else:
            self.onnx_bert_tokenize = create_op_function('BertTokenizer', bert_tokenize, hf_tok=self.hf_tok)

    def forward(self, text):
        return self.onnx_bert_tokenize(text)

    def export(self, *args, **kwargs):
        return _get_bound_object(self.onnx_bert_tokenize).build_model(kwargs.get('opset_version', 0), *args)


class PreHuggingFaceGPT2(ProcessingTracedModule):
    def __init__(self, hf_tok=None, vocab_file=None, merges_file=None, padding_length=-1):
        super(PreHuggingFaceGPT2, self).__init__()
        if hf_tok is None:
            self.onnx_gpt2_tokenize = create_op_function('GPT2Tokenizer', gpt2_tokenize,
                                                         vocab=_get_file_content(vocab_file),
                                                         merges=_get_file_content(merges_file),
                                                         padding_length=padding_length)
        else:
            self.onnx_gpt2_tokenize = create_op_function('GPT2Tokenizer', gpt2_tokenize, hf_tok=self.hf_tok)

    def forward(self, text):
        return self.onnx_gpt2_tokenize(text)

    def export(self, *args, **kwargs):
        return _get_bound_object(self.onnx_gpt2_tokenize).build_model(kwargs.get('opset_version', 0), *args)


