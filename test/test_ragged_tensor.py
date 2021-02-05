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
                ['tensor'], ['ragged'], domain=domain)
        ]
        input0 = helper.make_tensor_value_info('tensor', proto_type, None)
        output0 = helper.make_tensor_value_info(
            'ragged', onnx_proto.TensorProto.UINT8, [None])
        graph = helper.make_graph(nodes, 'test0', [input0], [output0])
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

            tensor = (np.arange(10) * 10).astype(dtype)
            txout = sess.run(None, {'tensor': tensor})

            if suffix == 'Float':
                expected = np.array([
                    1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 172, 0, 0, 0,
                    0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
                    0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
                    0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
                    7, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 9,
                    0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 32, 65, 0, 0, 160, 65, 0, 0, 240, 65,
                    0, 0, 32, 66, 0, 0, 72, 66, 0, 0, 112, 66, 0, 0,
                    140, 66, 0, 0, 160, 66, 0, 0, 180, 66], dtype=np.uint8)
                assert_almost_equal(expected, txout[0])

            tensor = (np.arange(30).reshape(3, -1) * 10).astype(dtype)
            txout = sess.run(None, {'tensor': tensor})
            if suffix == 'Float':
                expected = np.array([
                    1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 196, 0, 0, 0,
                    0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
                    0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0,
                    0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32,
                    65, 0, 0, 160, 65, 0, 0, 240, 65, 0, 0, 32, 66, 0, 0,
                    72, 66, 0, 0, 112, 66, 0, 0, 140, 66, 0, 0, 160, 66,
                    0, 0, 180, 66, 0, 0, 200, 66, 0, 0, 220, 66, 0, 0,
                    240, 66, 0, 0, 2, 67, 0, 0, 12, 67, 0, 0, 22, 67, 0,
                    0, 32, 67, 0, 0, 42, 67, 0, 0, 52, 67, 0, 0, 62, 67,
                    0, 0, 72, 67, 0, 0, 82, 67, 0, 0, 92, 67, 0, 0, 102,
                    67, 0, 0, 112, 67, 0, 0, 122, 67, 0, 0, 130, 67, 0,
                    0, 135, 67, 0, 0, 140, 67, 0, 0, 145, 67], dtype=np.uint8)
                assert_almost_equal(expected, txout[0])


if __name__ == "__main__":
    unittest.main()
