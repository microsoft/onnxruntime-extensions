import json
import onnx
import pathlib
import warnings
import numpy as np
from onnx import helper
from collections import namedtuple
from ._tensor import tensor_from_onnx, tensor_set_session
from ._onnx_ops import ONNXElementContainer, make_model_ex
from ..eager_op import SingleOpGraph, default_opset_domain, GPT2Tokenizer, VectorToString


def _is_path(name_or_buffer):
    return isinstance(name_or_buffer, str) or isinstance(name_or_buffer, pathlib.Path)


class ONNXTraceSession:
    activated_sessions = []

    def __init__(self, target_opset):
        self.container = ONNXElementContainer(target_opset)
        self.inputs = []
        self.outputs = []

    @classmethod
    def start_trace(cls, inputs, names=None, target_opset=11):
        self = ONNXTraceSession(target_opset)
        self.activated_sessions.append(self)
        tensor_set_session(self)

        np_inputs = [x if isinstance(x, (np.ndarray, np.generic)) else np.asarray(x) for x in inputs]
        itensors = [tensor_from_onnx(i_, None, None) for i_ in np_inputs]
        if names is not None:
            if len(inputs) != len(names):
                warnings.warn("the name number doesn't match the inputs', assign to the ones in the front.")
            num = min(len(itensors), len(names))
            for idx_ in range(num):
                itensors[idx_].name = names[idx_]
        self.inputs = itensors
        return itensors

    def __enter__(self):
        assert len(self.activated_sessions) > 0 and self.activated_sessions[-1] is self, "trace not started?"
        return self

    # need this exit to close the session
    def __exit__(self, exec_type, exec_value, exec_tb):
        tensor_set_session(None)
        assert self is self.activated_sessions.pop()

    @classmethod
    def stop_trace(cls, outputs):
        self = cls.get_active_session()
        self.set_outputs(outputs)
        return self

    @classmethod
    def get_active_session(cls):
        return cls.activated_sessions[0] if cls.activated_sessions else None

    def set_outputs(self, output_list):
        self.outputs = output_list

    def _reversed_travel(self):
        op_output_map = {}
        DynNode = namedtuple('DynNode', ['output'])
        input_node = DynNode(nm_.name for nm_ in self.inputs)
        for nd_ in self.container.nodes + [input_node]:
            for ky_ in nd_.output:
                op_output_map[ky_] = nd_

        active_nodes = [op_output_map[o_.name] for o_ in self.outputs]

        visited = {input_node}
        sorted_nodes = []
        while len(active_nodes) > 0:
            op_node = active_nodes.pop(0)
            if op_node.name in visited:
                continue

            sorted_nodes.insert(0, op_node)
            try:
                active_nodes.extend([op_output_map[o_] for o_ in op_node.input])
            except KeyError as e:
                raise RuntimeError("cannot find the operator to output {}".format(' '.join(op_node.input)))

        return sorted_nodes

    def build_model(self, model_name=None, doc_string=None) -> onnx.ModelProto:
        nodes = self._reversed_travel()
        container = self.container
        for tensor in self.container.initializers:
            # Initializers are always tensors so we can just call make_tensor_value_info(...)
            value_info = helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims)

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
    def serialize_attr(cls, kwargs):
        assert 'model' in kwargs, "Need model parameter to build the tokenizer"
        hf_gpt2_tokenizer = kwargs['model']
        attrs = {'vocab': json.dumps(hf_gpt2_tokenizer.encoder, separators=(',', ':'))}
        sorted_merges = {v_: k_ for k_, v_ in hf_gpt2_tokenizer.bpe_ranks.items()}
        attrs['merges'] = '\n'.join("{} {}".format(*sorted_merges[n_]) for n_ in range(len(sorted_merges)))
        return attrs


class _VectorToString(VectorToString):
    @classmethod
    def op_type(cls): return VectorToString.op_type()

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
    if _is_path(f):
        with open(f, 'wb') as f_:
            f_.write(m.SerializeToString())
    else:
        f.write(m.SerializeToString())
        f.flush()
