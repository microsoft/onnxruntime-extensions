# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
_ortapi2.py: ONNXRuntime-Extensions Python API
"""

import copy
import numpy as np
from ._ocos import default_opset_domain, get_library_path, Opdef
from ._cuops import onnx, onnx_proto, SingleOpGraph

_ort_check_passed = False
try:
    from packaging import version as _ver
    import onnxruntime as _ort

    if _ver.parse(_ort.__version__) >= _ver.parse("1.10.0"):
        _ort_check_passed = True
except ImportError:
    pass

if not _ort_check_passed:
    raise RuntimeError("please install ONNXRuntime/ONNXRuntime-GPU >= 1.10.0")


def _ensure_opset_domain(model):
    op_domain_name = default_opset_domain()
    domain_missing = True
    for oi_ in model.opset_import:
        if oi_.domain == op_domain_name:
            domain_missing = False

    if domain_missing:
        model.opset_import.extend([onnx.helper.make_operatorsetid(op_domain_name, 1)])

    return model


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
    brkpt_name = node_name + "_hkd"
    optype_name = "op_{}_{}".format(hook_func.__name__, node_name)
    for n_ in nodes:
        if n_.name == node_name:
            input_names = list(n_.input)
            brk_output_name = [i_ + "_hkd" for i_ in input_names]
            hnode = onnx.helper.make_node(
                optype_name, n_.input, brk_output_name, name=brkpt_name, domain=default_opset_domain()
            )
            nnode = n_
            del nnode.input[:]
            nnode.input.extend(brk_output_name)
            break
        n_idx += 1

    if hnode is None:
        raise ValueError("{} is not an operator node name".format(node_name))

    repacked = nodes[:n_idx] + [hnode, nnode] + nodes[n_idx + 1 :]
    del hkd_model.graph.node[:]
    hkd_model.graph.node.extend(repacked)

    Opdef.create(hook_func, op_type=optype_name, inputs=input_types, outputs=input_types)
    return _ensure_opset_domain(hkd_model)


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
    new_graph = onnx.helper.make_graph(new_nodes, graph.name, new_inputs, list(graph.output), list(graph.initializer))

    new_model = copy.deepcopy(model)
    new_model.graph.CopyFrom(new_graph)

    return _ensure_opset_domain(new_model)


def get_opset_version_from_ort():
    _ORT_OPSET_SUPPORT_TABLE = {
        "1.5": 11,
        "1.6": 12,
        "1.7": 13,
        "1.8": 14,
        "1.9": 15,
        "1.10": 15,
        "1.11": 16,
        "1.12": 17,
        "1.13": 17,
        "1.14": 18,
        "1.15": 18,
    }

    ort_ver_string = ".".join(_ort.__version__.split(".")[0:2])
    max_ver = max(_ORT_OPSET_SUPPORT_TABLE, key=_ORT_OPSET_SUPPORT_TABLE.get)
    if ort_ver_string > max_ver:
        ort_ver_string = max_ver
    return _ORT_OPSET_SUPPORT_TABLE.get(ort_ver_string, 11)


def make_onnx_model(graph, opset_version=0, extra_domain=default_opset_domain(), extra_opset_version=1):
    if opset_version == 0:
        opset_version = get_opset_version_from_ort()
    fn_mm = (
        onnx.helper.make_model_gen_version if hasattr(onnx.helper, "make_model_gen_version") else onnx.helper.make_model
    )
    model = fn_mm(graph, opset_imports=[onnx.helper.make_operatorsetid("ai.onnx", opset_version)])
    model.opset_import.extend([onnx.helper.make_operatorsetid(extra_domain, extra_opset_version)])
    return model


class OrtPyFunction:
    """
    OrtPyFunction is a convenience class that serves as a wrapper around the ONNXRuntime InferenceSession,
    equipped with registered onnxruntime-extensions. This allows execution of an ONNX model as if it were a
    standard Python function. The order of the function arguments correlates directly with
    the sequence of the input/output in the ONNX graph.
    """

    def get_ort_session_options(self):
        so = _ort.SessionOptions()
        for k, v in self.extra_session_options.items():
            so.__setattr__(k, v)
        so.register_custom_ops_library(get_library_path())
        return so

    def __init__(self, path_or_model=None, cpu_only=None):
        self._onnx_model = None
        self.ort_session = None
        self.default_inputs = {}
        self.execution_providers = ["CPUExecutionProvider"]
        if not cpu_only:
            if _ort.get_device() == "GPU":
                self.execution_providers = ["CUDAExecutionProvider"]
        self.extra_session_options = {}
        mpath = None
        if isinstance(path_or_model, str):
            oxml = onnx.load_model(path_or_model)
            mpath = path_or_model
        else:
            oxml = path_or_model
        if path_or_model is not None:
            self._bind(oxml, mpath)

    def create_from_customop(self, op_type, *args, **kwargs):
        graph = SingleOpGraph.build_graph(op_type, *args, **kwargs)
        self._bind(make_onnx_model(graph))
        return self

    def add_default_input(self, **kwargs):
        inputs = {
            ky_: val_ if isinstance(val_, (np.ndarray, np.generic)) else np.asarray(list(val_), dtype=np.uint8)
            for ky_, val_ in kwargs.items()
        }

        self.default_inputs.update(inputs)

    @property
    def onnx_model(self):
        assert self._oxml is not None, "No onnx model attached yet."
        return self._oxml

    @property
    def input_names(self):
        return [vi_.name for vi_ in self.onnx_model.graph.input]

    @property
    def output_names(self):
        return [vi_.name for vi_ in self.onnx_model.graph.output]

    def _bind(self, oxml, model_path=None):
        self.inputs = list(oxml.graph.input)
        self.outputs = list(oxml.graph.output)
        self._oxml = oxml
        if model_path is not None:
            self.ort_session = _ort.InferenceSession(
                model_path, self.get_ort_session_options(), self.execution_providers
            )
        return self

    def _ensure_ort_session(self):
        if self.ort_session is None:
            sess = _ort.InferenceSession(
                self.onnx_model.SerializeToString(), self.get_ort_session_options(), self.execution_providers
            )
            self.ort_session = sess

        return self.ort_session

    @staticmethod
    def _get_kwarg_device(kwargs):
        cpuonly = kwargs.get("cpu_only", None)
        if cpuonly is not None:
            del kwargs["cpu_only"]
        return cpuonly

    @classmethod
    def from_customop(cls, op_type, *args, **kwargs):
        return cls(cpu_only=cls._get_kwarg_device(kwargs)).create_from_customop(op_type, *args, **kwargs)

    @classmethod
    def from_model(cls, path_or_model, *args, **kwargs):
        fn = cls(path_or_model, cls._get_kwarg_device(kwargs))
        return fn

    def _argument_map(self, *args, **kwargs):
        idx = 0
        feed = {}
        for i_ in self.inputs:
            if i_.name in self.default_inputs:
                feed[i_.name] = self.default_inputs[i_.name]
                continue

            x = args[idx]
            ts_x = np.array(x) if isinstance(x, (int, float, bool)) else x
            # numpy by default is int32 in some platforms, sometimes it is int64.
            feed[i_.name] = (
                ts_x.astype(np.int64) if i_.type.tensor_type.elem_type == onnx_proto.TensorProto.INT64 else ts_x
            )
            idx += 1

        feed.update(kwargs)
        return feed

    def __call__(self, *args, **kwargs):
        self._ensure_ort_session()
        outputs = self.ort_session.run(None, self._argument_map(*args, **kwargs))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def ort_inference(model, *args, cpu_only=True, **kwargs):
    """
    Run an ONNX model with ORT where args are inputs and return values are outputs.
    """
    return OrtPyFunction(model, cpu_only=cpu_only)(*args, **kwargs)


def optimize_model(model_or_file, output_file):
    sess_options = OrtPyFunction().get_ort_session_options()
    sess_options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = output_file
    _ort.InferenceSession(
        model_or_file if isinstance(model_or_file, str) else model_or_file.SerializeToString(), sess_options
    )


ONNXRuntimeError = _ort.capi.onnxruntime_pybind11_state.Fail
ONNXRuntimeException = _ort.capi.onnxruntime_pybind11_state.RuntimeException
