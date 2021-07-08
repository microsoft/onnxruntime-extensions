import unittest
import numpy as np
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_extensions import get_library_path as _get_library_path


def _create_test_model(intput_dims, output_dims):
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node(
        'StringLength', ['identity1'], ['customout'], domain='ai.onnx.contrib')]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.STRING, [None] * intput_dims)
    output0 = helper.make_tensor_value_info(
        'customout', onnx_proto.TensorProto.INT64, [None] * output_dims)

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 12)])
    return model


def _run_string_length(input):
    model = _create_test_model(input.ndim, input.ndim)

    so = _ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
    sess = _ort.InferenceSession(model.SerializeToString(), so)
    result = sess.run(None, {'input_1': input})

    # verify
    output = np.array([len(elem) for elem in input.flatten()], dtype=np.int64).reshape(input.shape)
    np.testing.assert_array_equal(result, [output])


class TestStringLength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_vector_to_(self):
        _run_string_length(input=np.array(["a", "ab", "abc", "abcd"]))
        _run_string_length(input=np.array([" \t\n", ",.", "~!@#$%^&*()_+{}|:\"<>?[]\\;',./", "1234567890"]))
        _run_string_length(input=np.array([["we", "test", "whether"], ["it", "could", "output"], ["the", "same", "shape"]]))
        _run_string_length(input=np.array(["d(･｀ω´･d*)", "(σ｀д′)σ", "(;ﾟ∀ﾟ)=3ﾊｧﾊｧ", "(´థ౪థ）σ", "||ヽ(*￣▽￣*)ノミ|Ю"]))
        _run_string_length(input=np.array(["你", "好", "这是一个", "测试", ]))
        _run_string_length(input=np.array(["すみません、これはテストです。"]))
        _run_string_length(input=np.array(["Bonjour, c'est un test."]))
        _run_string_length(input=np.array(["Hallo, das ist ein Test."]))
        _run_string_length(input=np.array([" مرحبا هذا هو اختبار "]))
        _run_string_length(input=np.array(["👾 🤖 🎃 😺 😸 😹 😻 😼 😽 🙀 😿 😾"]))
        _run_string_length(input=np.array(["龖龘讋"]))


if __name__ == "__main__":
    unittest.main()
