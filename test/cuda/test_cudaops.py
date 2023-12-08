import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto
from onnxruntime_extensions import make_onnx_model
from onnxruntime_extensions import get_library_path as _get_library_path

import onnxruntime as _ort


class TestCudaOps(unittest.TestCase):
    @staticmethod
    def _create_negpos_test_model(domain='ai.onnx.contrib'):
        nodes = [
            helper.make_node('Identity', ['x'], ['identity1']),
            helper.make_node(
                'NegPos', ['identity1'], ['neg', 'pos'],
                domain=domain)
        ]

        input0 = helper.make_tensor_value_info(
            'x', onnx_proto.TensorProto.FLOAT, [None, None])
        output1 = helper.make_tensor_value_info(
            'neg', onnx_proto.TensorProto.FLOAT, [None, None])
        output2 = helper.make_tensor_value_info(
            'pos', onnx_proto.TensorProto.FLOAT, [None, None])

        graph = helper.make_graph(nodes, 'test0', [input0], [output1, output2])
        model = make_onnx_model(graph)
        return model

    def test_cuda_negpos(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_negpos_test_model()
        self.assertIn('op_type: "NegPos"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                     so,
                                     providers=['CUDAExecutionProvider'])
        x = np.array([[0., 1., 1.5], [7., 8., -5.5]]).astype(np.float32)
        neg, pos = sess.run(None, {'x': x})
        diff = x - (neg + pos)
        assert_almost_equal(diff, np.zeros(diff.shape))

    @staticmethod
    def _create_fastgelu_test_model(domain='ai.onnx.contrib'):
        nodes = [
            helper.make_node(
                'FastGelu', ['x', 'bias'], ['y'],
                domain=domain)
        ]

        input0 = helper.make_tensor_value_info(
            'x', onnx_proto.TensorProto.FLOAT, [])
        input1 = helper.make_tensor_value_info(
            'bias', onnx_proto.TensorProto.FLOAT, [])
        output0 = helper.make_tensor_value_info(
            'y', onnx_proto.TensorProto.FLOAT, [])

        graph = helper.make_graph(nodes, 'test1', [input0, input1], [output0])
        model = make_onnx_model(graph)
        return model

    def test_cuda_fastgelu(self):
        eps = _ort.get_available_providers()
        if 'CUDAExecutionProvider' in eps:
            so = _ort.SessionOptions()
            so.register_custom_ops_library(_get_library_path())
            onnx_model = self._create_fastgelu_test_model()
            self.assertIn('op_type: "FastGelu"', str(onnx_model))
            sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                         so,
                                         providers=['CUDAExecutionProvider'])
            x = np.array([0., 1., 2., 3., 4., 5.]).astype(np.float32)
            bias = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]).astype(np.float32)
            expected_y = np.array([0., 0.9505811, 2.1696784, 3.298689, 4.399991, 5.5]).astype(np.float32)
            y = sess.run(None, {'x': x, 'bias':bias})[0]
            assert_almost_equal(y, expected_y)
        else:
            print ('CUDAExecutionProvider not available, test_cuda_fastgelu skipped.')


if __name__ == "__main__":
    unittest.main()
