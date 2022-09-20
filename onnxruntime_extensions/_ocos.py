# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import sys
import copy
import onnx
from onnx import helper
from ._extensions_pydll import (  # noqa
    PyCustomOpDef, enable_py_op, add_custom_op, hash_64, default_opset_domain)


def get_library_path():
    """
    The custom operator library binary path
    :return: A string of the this library path.
    """
    mod = sys.modules['onnxruntime_extensions._extensions_pydll']
    return mod.__file__


class Opdef:

    _odlist = {}

    def __init__(self, op_type, func):
        self.op_type = op_type
        self.body = func
        self._id = id(self)

    @staticmethod
    def declare(*args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__call__'):
            raise RuntimeError("Unexpected arguments {}.".format(args))
            # return Opdef._create(args[0])
        return lambda f: Opdef.create(f, *args, **kwargs)

    @staticmethod
    def create(func, *args, **kwargs):
        name = kwargs.get('op_type', None)
        op_type = name or func.__name__
        opdef = Opdef(op_type, func)
        od_id = id(opdef)

        # Tells python this object cannot be destroyed
        # because it is also stored in C++ container.
        Opdef._odlist[od_id] = opdef
        opdef._nativedef = PyCustomOpDef()
        opdef._nativedef.op_type = op_type
        opdef._nativedef.obj_id = od_id

        inputs = kwargs.get('inputs', None)
        if inputs is None:
            inputs = [PyCustomOpDef.dt_float]
        opdef._nativedef.input_types = inputs
        outputs = kwargs.get('outputs', None)
        if outputs is None:
            outputs = [PyCustomOpDef.dt_float]
        opdef._nativedef.output_types = outputs
        attrs = kwargs.get('attrs', None)
        if attrs is None:
            attrs = []
        opdef._nativedef.attrs = attrs
        add_custom_op(opdef._nativedef)
        return opdef

    def __call__(self, *args, **kwargs):
        return self.body(*args, **kwargs)


def _on_pyop_invocation(k_id, feed, attributes):
    if k_id not in Opdef._odlist:
        raise RuntimeError(
            "Unable to find function id={}. "
            "Did you decorate the operator with @onnx_op?.".format(k_id))
    op_ = Opdef._odlist[k_id]
    rv = op_.body(*feed, **attributes)
    if isinstance(rv, tuple):
        # Multiple outputs.
        res = []
        for r in rv:
            res.append(r.shape)
            res.append(r.flatten().tolist())
        res = tuple(res)
    else:
        res = (rv.shape, rv.flatten().tolist())
    return (k_id, ) + res


def _ensure_opset_domain(model):
    op_domain_name = default_opset_domain()
    domain_missing = True
    for oi_ in model.opset_import:
        if oi_.domain == op_domain_name:
            domain_missing = False

    if domain_missing:
        model.opset_import.extend([helper.make_operatorsetid(op_domain_name, 1)])

    return model


def expand_onnx_inputs(model, target_input, extra_nodes, new_inputs):
    """
    Replace the existing inputs of a model with the new inputs, plus some extra nodes
    :param model: The ONNX model loaded as ModelProto
    :param target_input: The input name to be replaced
    :param extra_nodes: The extra nodes to be added
    :param new_inputs: The new input (type: ValueInfoProto) sequence
    :return: The ONNX model after modification
    """
    graph = model.graph
    new_inputs = [n for n in graph.input if n.name != target_input] + new_inputs
    new_nodes = list(model.graph.node) + extra_nodes
    new_graph = helper.make_graph(
        new_nodes, graph.name, new_inputs, list(graph.output), list(graph.initializer))

    new_model = copy.deepcopy(model)
    new_model.graph.CopyFrom(new_graph)

    return _ensure_opset_domain(new_model)


def hook_model_op(model, node_name, hook_func, input_types):
    """
    Add a hook function node in the ONNX Model, which could be used for the model diagnosis.
    :param model: The ONNX model loaded as ModelProto
    :param node_name: The node name where the hook will be installed
    :param hook_func: The hook function, callback on the model inference
    :param input_types: The input types as a list
    :return: The ONNX model with the hook installed
    """

    # onnx.shape_inference is very unstable, useless.
    # hkd_model = shape_inference.infer_shapes(model)
    hkd_model = model

    n_idx = 0
    hnode, nnode = (None, None)
    nodes = list(hkd_model.graph.node)
    brkpt_name = node_name + '_hkd'
    optype_name = "op_{}_{}".format(hook_func.__name__, node_name)
    for n_ in nodes:
        if n_.name == node_name:
            input_names = list(n_.input)
            brk_output_name = [i_ + '_hkd' for i_ in input_names]
            hnode = onnx.helper.make_node(
                optype_name, n_.input, brk_output_name, name=brkpt_name, domain=default_opset_domain())
            nnode = n_
            del nnode.input[:]
            nnode.input.extend(brk_output_name)
            break
        n_idx += 1

    if hnode is None:
        raise ValueError("{} is not an operator node name".format(node_name))

    repacked = nodes[:n_idx] + [hnode, nnode] + nodes[n_idx+1:]
    del hkd_model.graph.node[:]
    hkd_model.graph.node.extend(repacked)

    Opdef.create(hook_func, op_type=optype_name, inputs=input_types, outputs=input_types)
    return _ensure_opset_domain(hkd_model)


PyCustomOpDef.install_hooker(_on_pyop_invocation)
