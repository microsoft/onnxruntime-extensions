import copy
import onnx
import torch
import warnings
import numpy as np
from onnx import helper
from collections import namedtuple
from ._builder import is_path as _is_path
from ._onnx_ops import ONNXElementContainer, make_model_ex
from ._tensor import tensor_from_onnx, tensor_from_torch, tensor_set_session


class ONNXModelUtils:
    @staticmethod
    def _rename_iter(iterables, prefix_name, inplace=False):
        new_iz = iterables if inplace else [copy.deepcopy(iz_) for iz_ in iterables]
        for iz_ in new_iz:
            iz_.name = "{}_{}".format(prefix_name, iz_.name)
        return new_iz

    @classmethod
    def _rename_graph(cls, graph, prefix, graph_or_container):
        def io_rename(node, prefix_name):
            new_node = copy.deepcopy(node)
            del new_node.input[:]
            new_node.input.extend("{}_{}".format(prefix_name, nm_) for nm_ in node.input)
            del new_node.output[:]
            new_node.output.extend("{}_{}".format(prefix_name, nm_) for nm_ in node.output)
            return new_node

        assert prefix is not None, 'The graph prefix could not be None'
        graph_or_container.initializer.extend(cls._rename_iter(graph.initializer, prefix))
        graph_or_container.value_info.extend(cls._rename_iter(graph.value_info, prefix))
        return list(io_rename(nd_, prefix) for nd_ in graph.node)

    @classmethod
    def _process_node_body(cls, node, prefix):
        if all(attr.name != 'body' for attr in node.attribute):
            return node

        def _process_attr(attr, prefix_name):
            if attr.name == 'body':
                new_attr = copy.deepcopy(attr)
                del new_attr.g.value_info[:]
                del new_attr.g.node[:]
                new_attr.g.node.extend(cls._rename_graph(attr.g, prefix_name, new_attr.g))
                cls._rename_iter(new_attr.g.input, prefix_name, inplace=True)
                cls._rename_iter(new_attr.g.output, prefix_name, inplace=True)
                return new_attr
            else:
                return attr

        attr_list = list(_process_attr(attr_, prefix) for attr_ in node.attribute)
        del node.attribute[:]
        node.attribute.extend(attr_list)
        return node

    @classmethod
    def unfold_model_node(cls, container: ONNXElementContainer):
        nodes = container.nodes
        model_nodes = {node.name: node for node in nodes if hasattr(node, 'model')}
        onnx_nodes = [nd_ for nd_ in nodes if nd_.name not in model_nodes]

        for node in model_nodes.values():
            renamed_nodes = cls._rename_graph(node.model.graph, node.name, container)
            onnx_nodes.extend(cls._process_node_body(nd_, node.name) for nd_ in renamed_nodes)
        return onnx_nodes

    @classmethod
    def topological_sort(cls, container, nodes, inputs, outputs):
        op_output_map = {}
        DynNode = namedtuple('DynNode', ['name', 'output'])
        input_node = DynNode(name='placeholder',
                             output=[nm_.name for nm_ in inputs] +
                             [it_.name for it_ in container.initializers])
        for nd_ in nodes + [input_node]:
            for ky_ in nd_.output:
                op_output_map[ky_] = nd_

        edges = {}
        for op in nodes:
            for x in op.input:
                try:
                    predecessor = op_output_map[x]
                except KeyError:
                    raise RuntimeError("{}: cannot find the operator to produce {}".format(op.name, x))

                val = edges.get(predecessor.name, [])
                val.append(op)
                edges[predecessor.name] = val

        for y_ in outputs:
            op = op_output_map[y_.name].name
            if op not in edges:
                edges[op] = []

        visited = set()
        sorted_nodes = []
        unfinished_nodes = set()

        def recursive_helper(node):
            if node.name in visited:
                return

            if node.name in unfinished_nodes:
                raise RuntimeError("ONNX Graph is not a DAG, the cycle is found at {}".format(node.name))

            unfinished_nodes.add(node.name)
            if node.name in edges:  # if the node's output is not in the Graph output.
                for successor in edges[node.name]:
                    recursive_helper(successor)

            unfinished_nodes.remove(node.name)
            visited.add(node.name)
            sorted_nodes.insert(0, node)

        recursive_helper(input_node)
        assert sorted_nodes.pop(0) is input_node

        for op in nodes:  # non-input nodes
            if op.name not in visited:
                sorted_nodes.insert(0, op)

        return sorted_nodes


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

    def build_model(self, model_name=None, doc_string=None) -> onnx.ModelProto:
        container = self.container
        nodes = ONNXModelUtils.unfold_model_node(container)
        nodes = ONNXModelUtils.topological_sort(container, nodes, self.inputs, self.outputs)
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
