import json
import pathlib
from ._onnx_ops import make_model_ex
from .._cuops import SingleOpGraph, GPT2Tokenizer, VectorToString
from .._ortapi2 import default_opset_domain


def is_path(name_or_buffer):
    return isinstance(name_or_buffer, str) or isinstance(name_or_buffer, pathlib.Path)


class _GPT2Tokenizer(GPT2Tokenizer):
    @classmethod
    def serialize_attr(cls, kwargs):
        assert 'model' in kwargs, "Need model parameter to build the tokenizer"
        hf_gpt2_tokenizer = kwargs['model']
        attrs = {'vocab': json.dumps(hf_gpt2_tokenizer.encoder, separators=(',', ':'))}
        sorted_merges = {v_: k_ for k_, v_ in hf_gpt2_tokenizer.bpe_ranks.items()}
        attrs['merges'] = '\n'.join("{} {}".format(*sorted_merges[n_]) for n_ in range(len(sorted_merges)))
        return attrs


class _VectorToString(VectorToString):
    @classmethod
    def serialize_attr(cls, kwargs):
        assert 'decoder' in kwargs, "Need decoder parameter to build the tokenizer"
        decoder = kwargs['decoder']
        remapped = {v: [k] for k, v in decoder.items()}
        attrs = dict(map=remapped, unk='<unknown>')
        return super().serialize_attr(attrs)


customop_mbuilder = {
    c_.op_type(): c_ for c_ in (
        _GPT2Tokenizer,
        _VectorToString
    )
}


def build_customop_model(op_type, f, opset_version=11, **attrs):
    op_class = SingleOpGraph.get_op_class(op_type)
    if op_type in customop_mbuilder:
        op_class = customop_mbuilder[op_type]

    graph = SingleOpGraph.build_my_graph(op_class, **attrs)
    m = make_model_ex(graph, [(default_opset_domain(), 1)], opset_version)
    if is_path(f):
        with open(f, 'wb') as f_:
            f_.write(m.SerializeToString())
    else:
        f.write(m.SerializeToString())
        f.flush()
