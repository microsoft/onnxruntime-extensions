import onnx
import torch
import warnings
import numpy as np
from onnx import helper
from collections import namedtuple
from ._tensor import tensor_from_onnx, tensor_from_torch, tensor_set_session
from ._onnx_ops import ONNXElementContainer, make_model_ex
from ._builder import is_path as _is_path


class ONNXTraceSession:
    activated_sessions = []

    def __init__(self, target_opset):
        self.container = ONNXElementContainer(target_opset)
        self.inputs = []
        self.outputs = []

    def __enter__(self):
        assert len(self.activated_sessions) > 0 and self.activated_sessions[-1] is self, "trace not started?"
        return self

    # need this exit to close the session
    def __exit__(self, exec_type, exec_value, exec_tb):
        tensor_set_session(None)
        assert self is self.activated_sessions.pop()

    @classmethod
    def trace_for_onnx(cls, *inputs, names=None, target_opset=11) -> 'ONNXTraceSession':
        """
        Starting the trace all tensor computation for ONNX graph generation.
        :param inputs: the input tensor, could a torch.Tensor or a numpy ndarray.
        :param names: The input names the ONNX graph
        :param target_opset: The ONNX model opset_version
        :return: A tracing session object, in most case, it should be used in the with statement.
        """
        self = ONNXTraceSession(target_opset)
        self.activated_sessions.append(self)
        tensor_set_session(self)

        np_inputs = [x if isinstance(x, (np.ndarray, np.generic, torch.Tensor)) else np.asarray(x) for x in inputs]
        itensors = [tensor_from_torch(i_, None) if isinstance(i_, torch.Tensor)
                    else tensor_from_onnx(i_, None, None) for i_ in np_inputs]
        if names is not None:
            if len(inputs) != len(names):
                warnings.warn("the name number doesn't match the inputs', assign to the ones in the front.")
            num = min(len(itensors), len(names))
            for idx_ in range(num):
                itensors[idx_].name = names[idx_]
        self.inputs = itensors
        return self

    def get_inputs(self):
        return self.inputs

    @staticmethod
    def _unfold_model_node(container: ONNXElementContainer):
        nodes = container.nodes
        model_nodes = {node.name: node for node in nodes if hasattr(node, 'model')}
        onnx_nodes = [nd_ for nd_ in nodes if nd_.name not in model_nodes]
        for node in model_nodes.values():
            container.initializers.extend(list(node.model.graph.initializer))
            onnx_nodes.extend(node.model.graph.node)

        return onnx_nodes

    def _reversed_travel(self, container, nodes):
        op_output_map = {}
        DynNode = namedtuple('DynNode', ['name', 'output'])
        input_node = DynNode(name='placeholder',
                             output=[nm_.name for nm_ in self.inputs] +
                             [it_.name for it_ in container.initializers])
        for nd_ in nodes + [input_node]:
            for ky_ in nd_.output:
                op_output_map[ky_] = nd_

        active_nodes = [op_output_map[o_.name] for o_ in self.outputs]

        visited = {input_node.name}
        sorted_nodes = []
        while len(active_nodes) > 0:
            op_node = active_nodes.pop(0)
            if op_node.name in visited:
                continue

            visited.add(op_node.name)
            sorted_nodes.insert(0, op_node)
            try:
                active_nodes.extend([op_output_map[o_] for o_ in op_node.input])
            except KeyError as e:
                raise RuntimeError("{}: cannot find the operator to output {}".format(
                                    op_node.name, ' '.join(op_node.input)))
        return sorted_nodes

    def build_model(self, model_name=None, doc_string=None) -> onnx.ModelProto:
        container = self.container
        nodes = self._unfold_model_node(container)
        nodes = self._reversed_travel(container, nodes)
        model_name = 'tcm' if model_name is None else model_name
        doc_string = '' if doc_string is None else doc_string

        inputs = [helper.make_tensor_value_info(si.name, si.onnx_type,
                                                si.t.size()) for si in self.inputs]
        outputs = [helper.make_tensor_value_info(so.name, so.onnx_type,
                                                 so.t.size()) for so in self.outputs]

        graph = helper.make_graph(nodes, model_name, inputs,
                                  outputs, self.container.initializers)

        onnx_model = make_model_ex(graph, container.node_domain_version_pair_sets,
                                   container.target_opset, doc_string=doc_string)
        return onnx_model

    def save_as_onnx(self, file_like_or_path, outputs, model_name=None, doc_string=None):
        """
        Build the ONNX model from the traced computation graph.
        :param file_like_or_path: an io.BytesIO like object or a file path
        :param outputs: the output tensor to be specified as the ONNX graph output,
            Could be a string if there are multiple output tensors.
        :param model_name: The ONNX model internal name
        :param doc_string: The doc string for the model
        :return: A ONNX ModelProto object.
        """
        if len(self.outputs) == 0 and outputs is None:
            raise RuntimeError("No output of the graph specified.")

        if len(self.outputs) == 0:
            self.outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]

        m = self.build_model(model_name, doc_string)

        if _is_path(file_like_or_path):
            with open(file_like_or_path, 'wb') as f:
                f.write(m.SerializeToString())
        else:
            f = file_like_or_path
            f.write(m.SerializeToString())
            f.flush()

        return m
