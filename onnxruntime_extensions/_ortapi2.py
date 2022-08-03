# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import numpy as np
import onnxruntime as _ort
from ._ocos import default_opset_domain, get_library_path  # noqa
from ._cuops import *  # noqa


def get_opset_version_from_ort():
    _ORT_OPSET_SUPPORT_TABLE = {
        "1.5": 11,
        "1.6": 12,
        "1.7": 13,
        "1.8": 14,
        "1.9": 15,
        "1.10": 15,
        "1.11": 16,
        "1.12": 17
    }

    ort_ver_string = '.'.join(_ort.__version__.split('.')[0:2])
    return _ORT_OPSET_SUPPORT_TABLE.get(ort_ver_string, 11)


def make_onnx_model(graph, opset_version=0, extra_domain=default_opset_domain(), extra_opset_version=1):
    if opset_version == 0:
        opset_version = get_opset_version_from_ort()
    fn_mm = onnx.helper.make_model_gen_version if hasattr(onnx.helper, 'make_model_gen_version'
                                                          ) else onnx.helper.make_model
    model = fn_mm(graph, opset_imports=[
        onnx.helper.make_operatorsetid('ai.onnx', opset_version)])
    model.opset_import.extend([onnx.helper.make_operatorsetid(extra_domain, extra_opset_version)])
    return model


class OrtPyFunction:

    __name__ = 'OrtPyFunction'

    @classmethod
    def get_ort_session_options(cls):
        # ONNXRuntime has an issue to support reusing the SessionOptions object.
        # Create a new one every time here
        so = _ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        return so

    def __init__(self):
        self._onnx_model = None
        self.ort_session = None
        self.default_inputs = {}

    def create_from_customop(self, op_type, *args, **kwargs):
        graph = SingleOpGraph.build_my_graph(op_type, *args, **kwargs)
        self._bind(make_onnx_model(graph))
        return self

    def add_default_input(self, **kwargs):
        inputs = {
            ky_: val_ if isinstance(val_, (np.ndarray, np.generic)) else \
                np.asarray(list(val_), dtype=np.uint8) for ky_, val_ in kwargs.items()
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

    def _bind(self, oxml):
        self.inputs = list(oxml.graph.input)
        self.outputs = list(oxml.graph.output)
        self._oxml = oxml
        return self

    def _ensure_ort_session(self):
        if self.ort_session is None:
            sess = _ort.InferenceSession(self.onnx_model.SerializeToString(), self.get_ort_session_options())
            self.ort_session = sess

        return self.ort_session

    @classmethod
    def from_customop(cls, op_type, *args, **kwargs):
        return cls().create_from_customop(op_type, *args, **kwargs)

    @classmethod
    def from_model(cls, path_or_model, *args, **kwargs):
        return cls()._bind(onnx.load_model(path_or_model) if isinstance(path_or_model, str) else path_or_model)

    def _argument_map(self, *args, **kwargs):
        idx = 0
        feed = {}
        for i_ in self.inputs:
            if i_.name in self.default_inputs:
                feed[i_.name] = self.default_inputs[i_.name]
                continue

            x = args[idx]
            ts_x = np.array(x) if isinstance(x, (int, float, bool)) else x
            # an annoying bug is numpy by default is int32, while pytorch is int64.
            # so cast the input here automatically.
            feed[i_.name] = \
                ts_x.astype(np.int64) if i_.type.tensor_type.elem_type == onnx_proto.TensorProto.INT64 else ts_x
            idx += 1

        # feed.update(kwargs)
        return feed

    def __call__(self, *args, **kwargs):
        self._ensure_ort_session()
        outputs = self.ort_session.run(None, self._argument_map(*args, **kwargs))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def optimize_model(model_or_file, output_file):
    sess_options = OrtPyFunction.get_ort_session_options()
    sess_options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = output_file
    _ort.InferenceSession(model_or_file if isinstance(model_or_file, str)
                          else model_or_file.SerializeToString(), sess_options)


ONNXRuntimeError = _ort.capi.onnxruntime_pybind11_state.Fail
