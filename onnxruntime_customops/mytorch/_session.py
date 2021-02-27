import onnx
import pathlib
from onnx import helper, onnx_pb as onnx_proto
from ._tensor import Tensor
from ._onnx_ops import ONNXElementContainer, ONNXGraphBuilder, make_model_ex


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

    @staticmethod
    def _is_path(name_or_buffer):
        return isinstance(name_or_buffer, str) or isinstance(name_or_buffer, pathlib.Path)

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
        if self._is_path(file_like_or_path):
            f = open(file_like_or_path, 'wb')
        else:
            f = file_like_or_path

        m = self.build_model(model_name, doc_string)
        f.write(m.SerializeToString())


def _build_gpt2_tokenizer(hf_gpt2_tokenizer):
    pass


def _build_vector_to_string(id_dict):
    pass


customop_mbuilder = {
    'GPT2Tokenizer': _build_gpt2_tokenizer,
    'VectorToString': _build_vector_to_string
}


def build_customop_model(op_type, source, f, **attrs):
    pass
