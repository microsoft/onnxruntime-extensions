
import unittest
import numpy as np
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_customops import get_library_path as _get_library_path


def _create_test_model(map, unk, intput_dims, output_dims):
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node(
        'VectorToString', ['identity1'], ['customout'],
        map=_serialize_map(map), unk=unk,
        domain='ai.onnx.contrib')]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.INT64, [None] * intput_dims)
    output0 = helper.make_tensor_value_info(
        'customout', onnx_proto.TensorProto.STRING, [None] * output_dims)

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 12)])
    return model


def _serialize_map(map):
    result = ""
    for k in map:
        result = result + k + "\t" + " ".join([str(i) for i in map[k]]) + "\n"
    return result


def _run_vector_to_string(input, output, map, unk):
    model = _create_test_model(map, unk, input.ndim, input.ndim)

    so = _ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
    sess = _ort.InferenceSession(model.SerializeToString(), so)
    result = sess.run(None, {'input_1': input})
    np.testing.assert_array_equal(result, [output])


class TestVectorToString(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_vector_to_(self):
        _run_vector_to_string(
            input=np.array([0, 2, 3, 4], dtype=np.int64),
            output=np.array(["a", "b", "c", "unknown_word"]),
            map={"a": [0], "b": [2], "c": [3]},
            unk="unknown_word")

        _run_vector_to_string(
            input=np.array([[0, ], [2, ], [3, ], [4, ]], dtype=np.int64),
            output=np.array(["a", "b", "c", "unknown_word"]),
            map={"a": [0], "b": [2], "c": [3]},
            unk="unknown_word")

        _run_vector_to_string(
            input=np.array([[0, 1], [2, 3], [3, 4], [4, 5]], dtype=np.int64),
            output=np.array(["a", "b", "c", "unknown_word"]),
            map={"a": [0, 1], "b": [2, 3], "c": [3, 4]},
            unk="unknown_word")


if __name__ == "__main__":
    unittest.main()
