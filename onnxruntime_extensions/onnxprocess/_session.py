import copy
import onnx
import torch
import warnings
import numpy as np
from onnx import helper, mapping
from collections import namedtuple
from .._ortapi2 import OrtPyFunction
from ._builder import is_path as _is_path
from ._onnx_ops import ONNXElementContainer, make_model_ex
from ._tensor import tensor_from_onnx, tensor_from_torch, tensor_set_session


def _is_numpy_object(x):
    return isinstance(x, (np.ndarray, np.generic))


def _is_numpy_string_type(arr):
    return arr.dtype.kind in {'U', 'S'}


def _is_string_type(x):
    if not _is_numpy_object(x):
        x = np.array(x)
    return _is_numpy_string_type(x)


class ONNXModelUtils:
    @staticmethod
    def _rename_iter(iterables, prefix_name, inplace=False):
        new_iz = iterables if inplace else [copy.deepcopy(iz_) for iz_ in iterables]
        for iz_ in new_iz:
            iz_.name = "{}_{}".format(prefix_name, iz_.name)
        return new_iz

    @classmethod
    def _rename_graph(cls, graph, prefix, graph_or_container):
        def io_rename(node, prefix_name, idx):
            new_node = copy.deepcopy(node)
            if not node.name:
                new_node.name = "{}_op{}".format(prefix_name, idx)

            del new_node.input[:]
            new_node.input.extend("{}_{}".format(prefix_name, nm_) if nm_ else '' for nm_ in node.input)
            del new_node.output[:]
            new_node.output.extend("{}_{}".format(prefix_name, nm_) if nm_ else '' for nm_ in node.output)
            return new_node

        assert prefix is not None, 'The graph prefix could not be None'
        graph_or_container.initializer.extend(cls._rename_iter(graph.initializer, prefix))
        graph_or_container.value_info.extend(cls._rename_iter(graph.value_info, prefix))
        return list(io_rename(nd_, prefix, idx_) for idx_, nd_ in enumerate(graph.node))

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
        top_containter = container
        while top_containter.parent is not None:  # only one opset_import in the model.
            top_containter = top_containter.parent

        nodes = container.nodes
        model_nodes = {node.name: node for node in nodes if hasattr(node, 'model')}
        onnx_nodes = [nd_ for nd_ in nodes if nd_.name not in model_nodes]

        for node in model_nodes.values():
            renamed_nodes = cls._rename_graph(node.model.graph, node.name, container)
            onnx_nodes.extend(cls._process_node_body(nd_, node.name) for nd_ in renamed_nodes)

            top_containter.node_domain_version_pair_sets.update([(opset_.domain, opset_.version) for opset_ in node.model.opset_import])
        return onnx_nodes

    @classmethod
    def topological_sort(cls, container, nodes, inputs, outputs):
        op_output_map = {}
        DynNode = namedtuple('DynNode', ['name', 'output'])
        input_nodes = [DynNode(name='placeholder',
                               output=[nm_.name for nm_ in inputs] +
                                      [it_.name for it_ in container.initializers])] +\
                      [nd_ for nd_ in nodes if nd_.op_type == 'Constant']

        for nd_ in nodes + input_nodes:
            for ky_ in nd_.output:
                op_output_map[ky_] = nd_

        edges = {}
        for op in nodes:
            for x in op.input:
                if x == '':
                    continue
                try:
                    predecessor = op_output_map[x]
                except KeyError:
                    raise RuntimeError(
                        "{}: cannot find an operator to produce the tensor: {}".format(op.name, x)) from None

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
                assert node.name != '', 'this topological-sort depends on the unique node name.'
                for successor in edges[node.name]:
                    recursive_helper(successor)

            unfinished_nodes.remove(node.name)
            visited.add(node.name)
            if node is not input_nodes[0]:
                sorted_nodes.insert(0, node)

        for nd_ in input_nodes:
            recursive_helper(nd_)

        return sorted_nodes

    @staticmethod
    def value_info_from_numpy(name, value):
        dtype = onnx.onnx_pb.TensorProto.STRING if \
            _is_numpy_string_type(value) else mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype]
        return helper.make_tensor_value_info(name, dtype, shape=value.shape)

    @staticmethod
    def model_from_ops(container, ops, ts_from, ts_to):
        all_inputs = []
        all_outputs = []
        iz_needed = set()
        iz_set = set(iz_.name for iz_ in container.initializer)
        for op in ops:
            iz_needed.update(it_ for it_ in op.input if it_ in iz_set)
            all_inputs.extend(it_ for it_ in op.input if (it_ != '') and it_ not in iz_set)
            all_outputs.extend(ot_ for ot_ in op.output)

        intersections = set(all_inputs).intersection(set(all_outputs))
        assert set(all_inputs).difference(intersections) == set(ts_.name for ts_ in ts_from), \
            "The input list is different from the calculated from the op nodes"
        assert set(all_outputs).difference(intersections) == set(ts_.name for ts_ in ts_to), \
            "The output list is different from the calculated from the op nodes"

        final_iz = [iz_ for iz_ in container.initializers if iz_.name in iz_needed]
        graph = helper.make_graph(ops, 'dyngraph', ts_from, ts_to, final_iz)
        oxml = make_model_ex(graph,
                             container.node_domain_version_pair_sets,
                             container.target_opset)
        return oxml


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

        np_inputs = [np.array(x) if _is_string_type(x) else x for x in inputs]
        np_inputs = [
            x if isinstance(x, (np.ndarray, np.generic, torch.Tensor)) or _is_string_type(x)
            else torch.tensor(x) for x in np_inputs]
        itensors = [tensor_from_torch(i_, None) if isinstance(i_, torch.Tensor)
                    else tensor_from_onnx(i_, None, None) for i_ in np_inputs]
        if names is None:
            names = []
        if len(inputs) != len(names):
            warnings.warn("the name number doesn't match the inputs', assign to the ones in the front.")
            names.extend([''] * (len(inputs) - len(names)))
            for idx_ in range(len(inputs)):
                names[idx_] = names[idx_] if names[idx_] else "input{}".format(idx_)
        num = min(len(itensors), len(names))
        for idx_ in range(num):
            itensors[idx_].name = names[idx_]
        self.inputs = itensors
        return self

    def runops(self, ts_from, ts_to):
        nodes = self.container.nodes
        inset = set(ts_.name for ts_ in ts_from)
        inset.update(iz_.name for iz_ in self.container.initializer)
        outset = set(ts_.name for ts_ in ts_to)
        missing_ts_set = set()
        node_num = len(nodes) - 1
        while node_num >= 0:
            node = nodes[node_num]
            for ot_ in node.output:
                if ot_ in missing_ts_set:
                    missing_ts_set.remove(ot_)
                elif ot_ in outset:
                    outset.remove(ot_)
            for it_ in node.input:
                if it_ not in inset:
                    missing_ts_set.add(it_)
            if len(missing_ts_set) == 0:
                break
            node_num -= 1

        assert len(outset) == 0, "Some output cannot be in the node list."
        assert len(missing_ts_set) == 0, "Some input cannot be in the node list."
        collected_nodes = nodes[node_num:]
        vi_input = [ONNXModelUtils.value_info_from_numpy(ts_.name, ts_.numpy())
                    for ts_ in ts_from]
        vi_output = [ONNXModelUtils.value_info_from_numpy(ts_.name, ts_.numpy())
                     for ts_ in ts_to]
        oxml = ONNXModelUtils.model_from_ops(self.container,
                                             collected_nodes,
                                             vi_input,
                                             vi_output)
        result = None
        try:
            oxfunc = OrtPyFunction.from_model(oxml)
            result = oxfunc(*[ts_.numpy() for ts_ in ts_from])
        finally:
            if result is None:
                onnx.save_model(oxml, 'mt_debmodel.onnx')

        return result if isinstance(result, (list, tuple)) else [result], oxml

    def get_inputs(self):
        return self.inputs

    def stack_container(self):
        assert self.container is not None, "Stacked container must be in another one."
        sub_container = ONNXElementContainer(self.container.target_opset, self.container)
        self.container = sub_container
        return self.container

    def pop_container(self):
        assert self.container.parent is not None, "Cannot pop the root container."
        self.container = self.container.parent
        return self.container

    @staticmethod
    def build_graph(container, ts_inputs, ts_outputs, graph_name=None):
        # some constant ops are created to simulate the tensors generated from the runtime in the loop,
        # so we need to remove the node here
        to_del = []
        input_names = {it_.name: None for it_ in ts_inputs}
        for idx_, nd_ in enumerate(container.nodes):
            if nd_.op_type == 'Constant' and list(nd_.output)[0] in input_names:
                to_del.append(idx_)

        for idx_ in to_del[::-1]:
            container.nodes.pop(idx_)

        graph_name = container.get_unique_operator_name('subg') if not graph_name else graph_name
        nodes = ONNXModelUtils.unfold_model_node(container)
        nodes = ONNXModelUtils.topological_sort(container, nodes, ts_inputs, ts_outputs)

        for vi_ in container.value_info:
            if vi_.name in input_names:
                input_names[vi_.name] = vi_

        inputs = [helper.make_tensor_value_info(si.name, si.onnx_type, si.get_shape())
                  if input_names.get(si.name) is None else input_names[si.name] for si in ts_inputs]
        outputs = [helper.make_tensor_value_info(so.name, so.onnx_type,
                                                 so.get_shape()) for so in ts_outputs]

        graph = helper.make_graph(nodes, graph_name, inputs,
                                  outputs, container.initializers)
        return graph

    def build_model(self, model_name=None, doc_string=None) -> onnx.ModelProto:
        model_name = 'tcm' if model_name is None else model_name
        doc_string = '' if doc_string is None else doc_string
        container = self.container
        graph = self.build_graph(container, self.inputs, self.outputs, model_name)
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

        if file_like_or_path is not None:
            if _is_path(file_like_or_path):
                with open(file_like_or_path, 'wb') as f:
                    f.write(m.SerializeToString())
            else:
                f = file_like_or_path
                f.write(m.SerializeToString())
                f.flush()

        return m
