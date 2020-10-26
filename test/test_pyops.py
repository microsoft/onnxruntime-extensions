import unittest
import os
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto, load
import onnxruntime as _ort
from ortcustomops import (
    onnx_op,
    get_library_path as _get_library_path)


def _create_test_model():
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('ReverseMatrix',
                                  ['identity1'], ['reversed'],
                                  domain='ai.onnx.contrib')]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.FLOAT, [None, 2])
    output0 = helper.make_tensor_value_info(
        'reversed', onnx_proto.TensorProto.FLOAT, [None, 2])

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid('ai.onnx.contrib', 1)])
    return model


@onnx_op(op_type="ReverseMatrix")
def reverse_matrix(x):
    # The user custom op implementation here.
    return np.flip(x, axis=0).astype(np.float32)


class TestPythonOp(unittest.TestCase):

    def test_python_operator(self):

        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model()
        self.assertIn('op_type: "ReverseMatrix"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array(
            [1, 2, 3, 4, 5, 6]).astype(np.float32).reshape([3, 2])
        txout = sess.run(None, {'input_1': input_1})
        assert_almost_equal(txout[0], np.array([[5., 6.], [3., 4.], [1., 2.]]))

    def test_cc_operator(self):

        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())

        this = os.path.dirname(__file__)
        filename = os.path.join(this, 'data', 'custom_op_test.onnx')
        onnx_content = load(filename)
        self.assertIn('op_type: "CustomOpOne"', str(onnx_content))
        sess0 = _ort.InferenceSession(filename, so)

        res = sess0.run(None, {
            'input_1': np.random.rand(3, 5).astype(np.float32),
            'input_2': np.random.rand(3, 5).astype(np.float32)})

        self.assertEqual(res[0].shape, (3, 5))


if __name__ == "__main__":
    unittest.main()
