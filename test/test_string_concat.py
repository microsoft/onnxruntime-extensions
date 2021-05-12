import os
from pathlib import Path
import unittest
import numpy as np
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_extensions import (
    onnx_op,
    enable_custom_op,
    PyCustomOpDef,
    expand_onnx_inputs,
    get_library_path as _get_library_path)


def _create_test_model(input_dims, output_dims):
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['left'])]
    nodes[1:] = [helper.make_node('Identity', ['input_2'], ['right'])]
    nodes[2:] = [helper.make_node(
        'StringConcat', ['left', 'right'], ['output'], domain='ai.onnx.contrib')]

    input1 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.STRING, [None] * input_dims)
    input2 = helper.make_tensor_value_info(
        'input_2', onnx_proto.TensorProto.STRING, [None] * input_dims)
    output = helper.make_tensor_value_info(
        'output', onnx_proto.TensorProto.STRING, [None] * output_dims)

    graph = helper.make_graph(nodes, 'test0', [input1, input2], [output])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 12)])
    return model


def _run_string_concat(input1, input2):
    model = _create_test_model(input1.ndim, input1.ndim)

    so = _ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
    sess = _ort.InferenceSession(model.SerializeToString(), so)
    result = sess.run(None, {'input_1': input1, 'input_2': input2})

    # verify
    output = []
    shape = input1.shape
    input1 = input1.flatten()
    input2 = input2.flatten()
    for i in range(len(input1)):
        output.append(input1[i] + input2[i])
    output = np.array(output).reshape(shape)
    np.testing.assert_array_equal(result, [output])


class TestStringConcat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_string_concat(self):
        _run_string_concat(np.array(["a"]), np.array(["b"]))
        _run_string_concat(np.array(["a", "b", "c", "d"]), np.array(["d", "c", "b", "a"]))
        _run_string_concat(np.array([["a", "b"], ["c", "d"]]), np.array([["d", "c"], ["b", "a"]]))
        _run_string_concat(np.array(["你好"]), np.array(["这是一个测试"]))
        _run_string_concat(np.array(["すみません"]), np.array(["これはテストです"]))
        _run_string_concat(np.array(["👾 🤖 🎃 😺 😸 😹"]), np.array(["😻 😼 😽 🙀 😿 😾"]))
        _run_string_concat(np.array(["龖"]), np.array(["龘讋"]))


if __name__ == "__main__":
    unittest.main()
