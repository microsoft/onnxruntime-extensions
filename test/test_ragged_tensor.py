# coding: utf-8
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_customops import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path)


class TestPythonOpRaggedTensor(unittest.TestCase):

    def enumerate_proto_types(self):
        for p in [(onnx_proto.TensorProto.FLOAT, np.float32, 'Float'),
                  (onnx_proto.TensorProto.DOUBLE, np.float64, 'Double'),
                  (onnx_proto.TensorProto.INT64, np.int64, 'Int64')]:
            with self.subTest(type=p[2]):
                yield p

    def _create_test_model_tensor_to_ragged_tensor(
            self, prefix, proto_type, suffix, domain='ai.onnx.contrib'):
        nodes = [
            helper.make_node(
                '%sTensorToRaggedTensor%s' % (prefix, suffix),
                ['tensor'], ['values', 'indices'], domain=domain)
        ]
        input0 = helper.make_tensor_value_info('tensor', proto_type, None)
        output0 = helper.make_tensor_value_info('values', proto_type, [None])
        output1 = helper.make_tensor_value_info(
            'indices', onnx_proto.TensorProto.INT64, [None])
        graph = helper.make_graph(nodes, 'test0', [input0], [output0, output1])
        model = helper.make_model(
            graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
        return model

    @classmethod
    def setUpClass(cls):

        """
        @onnx_op(op_type="PyStringUpper",
                 inputs=[PyCustomOpDef.dt_string],
                 outputs=[PyCustomOpDef.dt_string])
        def string_upper(x):
            # The user custom op implementation here.
            return np.array([s.upper() for s in x.ravel()]).reshape(x.shape)
        """

        pass

    def test_tensor_to_ragged_tensor(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        for proto_type, dtype, suffix in self.enumerate_proto_types():
            onnx_model = self._create_test_model_tensor_to_ragged_tensor(
                prefix='', proto_type=proto_type, suffix=suffix)
            self.assertIn('op_type: "TensorToRaggedTensor%s"' % suffix,
                          str(onnx_model))
            sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)

            tensor = (np.random.randn(10) * 10).astype(dtype)
            txout = sess.run(None, {'tensor': tensor})
            assert_almost_equal(tensor.ravel(), txout[0].ravel())
            assert_almost_equal(np.array([0, 10], dtype=np.int64),
                                txout[1].ravel())

            tensor = (np.random.randn(3, 10) * 10).astype(dtype)
            txout = sess.run(None, {'tensor': tensor})
            assert_almost_equal(tensor.ravel(), txout[0].ravel())
            assert_almost_equal(np.array([0, 10, 20, 30], dtype=np.int64),
                                txout[1].ravel())


if __name__ == "__main__":
    unittest.main()
