import onnx
import pathlib
from onnx import helper, onnx_pb as onnx_proto
from ._tensor import Tensor
from ._onnx_ops import ONNXElementContainer, ONNXGraphBuilder, make_model_ex

from ..eager_op import SingleOpGraph, default_opset_domain, GPT2Tokenizer, VectorToString


def _is_path(name_or_buffer):
    return isinstance(name_or_buffer, str) or isinstance(name_or_buffer, pathlib.Path)


class ONNXTraceSession:
    activated_sessions = []

    def __init__(self, inputs, target_opset):
        self.container = ONNXElementContainer(target_opset)
        self.ox = ONNXGraphBuilder()
        self.inputs = inputs
        self.torch_ops = []

    def __enter__(self):
        self.activated_sessions.append(self)
        Tensor.set_active_session(self)
        return self

    def __exit__(self, exec_type, exec_value, exec_tb):
        Tensor.set_active_session(None)
        last = self.activated_sessions[-1]
        del self.activated_sessions[-1:]
        return last

    @classmethod
    def get_active_session(cls):
        return cls.activated_sessions[0] if cls.activated_sessions else None

    def set_outputs(self, output_list):
        pass

    def _topology_sort(self):
        pass

    def stop_trace(self, outputs):
        self.set_outputs(outputs)

    def build_model(self, model_name=None, doc_string=None) -> onnx.ModelProto:
        self._topology_sort()
        container = self.container
        for tensor in self.container.initializers:
            # Initializers are always tensors so we can just call make_tensor_value_info(...)
            value_info = helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims)

        nodes = self.container.nodes
        graph = helper.make_graph(nodes, model_name, self.container.inputs,
                                  self.container.outputs, self.container.initializers)

        # Add extra information related to the graph
        graph.value_info.extend(self.container.value_info)
        onnx_model = make_model_ex(graph, container.node_domain_version_pair_sets,
                                   container.target_opset, doc_string=doc_string)
        return onnx_model

    def save_as_onnx(self, file_like_or_path, model_name=None, doc_string=None):
        """
        Build the ONNX model from the traced computation graph.
        :param file_like_or_path:
        :param model_name:
        :param doc_string:
        :return:
        """
        if _is_path(file_like_or_path):
            f = open(file_like_or_path, 'wb')
        else:
            f = file_like_or_path

        m = self.build_model(model_name, doc_string)
        f.write(m.SerializeToString())


class _GPT2Tokenizer(GPT2Tokenizer):
    @classmethod
    def op_type(cls): return GPT2Tokenizer.op_type()

    @classmethod
    def build(cls, hf_gpt2_tokenizer, **attrs):
        import json
        attrs['vocab'] = json.dumps(hf_gpt2_tokenizer.encoder)
        sorted_merges = {v_: k_ for k_, v_ in hf_gpt2_tokenizer.bpe_ranks.items()}
        attrs['merges'] = ',\n'.join("{} {}".format(*sorted_merges[n_]) for n_ in range(len(sorted_merges)))
        return SingleOpGraph.build_singleop_graph(cls.op_type(), **attrs)


class _VectorToString(VectorToString):
    @classmethod
    def op_type(cls): return VectorToString.op_type()

    @classmethod
    def build(cls, id_dict, **attrs):
        remap = {v: [k] for k,v in id_dict.items()}
        attrs['map'] = '\n'.join(k + "\t" + " ".join([str(i) for i in v]) for k, v in remap.items())
        return SingleOpGraph.build_singleop_graph(cls.op_type(), **attrs)


customop_mbuilder = {
    c_.op_type(): c_ for c_ in (
        _GPT2Tokenizer,
        _VectorToString
    )
}


def build_customop_model(op_type, source, f, opset_version=11, **attrs):
    if op_type in customop_mbuilder:
        graph = customop_mbuilder[op_type].build(source, **attrs)
    else:
        graph = SingleOpGraph.build_singleop_graph(op_type, **attrs)

    m = make_model_ex(graph, [(default_opset_domain(), 1)], opset_version)
    if _is_path(f):
        f = open(f, 'wb')
    f.write(m.SerializeToString())
    f.flush()
