import os
from pathlib import Path
import unittest
import numpy as np
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_customops import (
    onnx_op,
    enable_custom_op,
    PyCustomOpDef,
    expand_onnx_inputs,
    get_library_path as _get_library_path)


def _create_test_model(map, unk):
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node(
        'VectorToString', ['identity1'], ['customout'], map=_serialize_map(map), unk=unk,
        domain='ai.onnx.contrib')]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.INT64, [None])
    output0 = helper.make_tensor_value_info(
        'customout', onnx_proto.TensorProto.STRING, [None])

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 12)])
    return model


def _serialize_map(map):
    result = ""
    for k in map:
        result = result + k + "\t" + " ".join([str(i) for i in map[k]]) + "\n"
    return result


def _run_vector_to_string(input, map, unk):
    model = _create_test_model(map, unk)

    so = _ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
    sess = _ort.InferenceSession(model.SerializeToString(), so)
    return sess.run(None, {'input_1': input})


class TestGPT2Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_vector_to_(self):
        result = _run_vector_to_string(input=np.array([0, 2, 3, 4], dtype=np.int64), map={"a": [0], "b": [2], "c": [3]}, unk="unknown_word")
        np.testing.assert_array_equal(result, [np.array(["a", "b", "c", "unknown_word"])])


if __name__ == "__main__":
    unittest.main()
