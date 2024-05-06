import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, numpy_helper, onnx_pb as onnx_proto, TensorProto
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from onnxruntime_extensions import make_onnx_model
from onnxruntime_extensions import get_library_path as _get_library_path

import onnxruntime as _ort


class NegXPlus1(OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, X):
        return (1 - X,)


class TestCudaOps(unittest.TestCase):
    @staticmethod
    def _create_negpos_test_model(domain="ai.onnx.contrib"):
        nodes = [
            helper.make_node("Identity", ["x"], ["identity1"]),
            helper.make_node("NegPos", ["identity1"], ["neg", "pos"], domain=domain),
        ]

        input0 = helper.make_tensor_value_info("x", onnx_proto.TensorProto.FLOAT, [None, None])
        output1 = helper.make_tensor_value_info("neg", onnx_proto.TensorProto.FLOAT, [None, None])
        output2 = helper.make_tensor_value_info("pos", onnx_proto.TensorProto.FLOAT, [None, None])

        graph = helper.make_graph(nodes, "test0", [input0], [output1, output2])
        model = make_onnx_model(graph)
        return model

    def test_cuda_negpos(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_negpos_test_model()
        self.assertIn('op_type: "NegPos"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so, providers=["CUDAExecutionProvider"])
        x = np.array([[0.0, 1.0, 1.5], [7.0, 8.0, -5.5]]).astype(np.float32)
        neg, pos = sess.run(None, {"x": x})
        diff = x - (neg + pos)
        assert_almost_equal(diff, np.zeros(diff.shape))

    @staticmethod
    def _create_fastgelu_test_model(domain="ai.onnx.contrib"):
        nodes = [helper.make_node("FastGelu", ["x", "bias"], ["y"], domain=domain)]

        input0 = helper.make_tensor_value_info("x", onnx_proto.TensorProto.FLOAT, [])
        input1 = helper.make_tensor_value_info("bias", onnx_proto.TensorProto.FLOAT, [])
        output0 = helper.make_tensor_value_info("y", onnx_proto.TensorProto.FLOAT, [])

        graph = helper.make_graph(nodes, "test1", [input0, input1], [output0])
        model = make_onnx_model(graph)
        return model

    @staticmethod
    def _create_fastgelu_test_model_f16(domain="ai.onnx.contrib"):
        nodes = [helper.make_node("FastGelu", ["x", "bias"], ["y"], domain=domain)]

        input0 = helper.make_tensor_value_info("x", onnx_proto.TensorProto.FLOAT16, [])
        input1 = helper.make_tensor_value_info("bias", onnx_proto.TensorProto.FLOAT16, [])
        output0 = helper.make_tensor_value_info("y", onnx_proto.TensorProto.FLOAT16, [])

        graph = helper.make_graph(nodes, "test1", [input0, input1], [output0])
        model = make_onnx_model(graph)
        return model

    def test_cuda_fastgelu(self):
        eps = _ort.get_available_providers()
        if "CUDAExecutionProvider" in eps:
            so = _ort.SessionOptions()
            so.register_custom_ops_library(_get_library_path())
            onnx_model = self._create_fastgelu_test_model()
            self.assertIn('op_type: "FastGelu"', str(onnx_model))
            sess = _ort.InferenceSession(onnx_model.SerializeToString(), so, providers=["CUDAExecutionProvider"])
            x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
            bias = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]).astype(np.float32)
            expected_y = np.array([0.0, 0.9505811, 2.1696784, 3.298689, 4.399991, 5.5]).astype(np.float32)
            y = sess.run(None, {"x": x, "bias": bias})[0]
            assert_almost_equal(y, expected_y)
        else:
            print("CUDAExecutionProvider not available, test_cuda_fastgelu skipped.")

    def test_cuda_fastgelu_f16(self):
        eps = _ort.get_available_providers()
        if "CUDAExecutionProvider" in eps:
            so = _ort.SessionOptions()
            so.register_custom_ops_library(_get_library_path())
            onnx_model = self._create_fastgelu_test_model_f16()
            self.assertIn('op_type: "FastGelu"', str(onnx_model))
            sess = _ort.InferenceSession(onnx_model.SerializeToString(), so, providers=["CUDAExecutionProvider"])
            x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float16)
            bias = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]).astype(np.float16)
            expected_y = np.array([0.0, 0.95, 2.17, 3.299, 4.4, 5.5]).astype(np.float16)
            y = sess.run(None, {"x": x, "bias": bias})[0]
            assert_almost_equal(y, expected_y)
        else:
            print("CUDAExecutionProvider not available, test_cuda_fastgelu_f16 skipped.")

    def _negxplus1_cuda(self, itype):
        import onnxruntime

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        model1 = helper.make_model(
            helper.make_graph(
                [helper.make_node("Sub", ["one", "X"], ["Y"])],
                "nd",
                [helper.make_tensor_value_info("X", itype, [None, None, None])],
                [helper.make_tensor_value_info("Y", itype, [None, None, None])],
                [numpy_helper.from_array(np.array([1], dtype=dtype), name="one")],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = helper.make_model(
            helper.make_graph(
                [helper.make_node("NegXPlus1", ["X"], ["Y"], domain="ai.onnx.contrib")],
                "nd",
                [helper.make_tensor_value_info("X", itype, [None, None, None])],
                [helper.make_tensor_value_info("Y", itype, [None, None, None])],
            ),
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("ai.onnx.contrib", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        x = (np.arange(18) - 4).reshape((3, 2, 3)).astype(dtype)

        feeds1 = dict(X=x)
        ref = ReferenceEvaluator(model1, new_ops=[NegXPlus1])
        expected = ref.run(None, feeds1)[0]

        opts = onnxruntime.SessionOptions()
        opts.register_custom_ops_library(_get_library_path())
        sess = onnxruntime.InferenceSession(model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"])
        got = sess.run(None, feeds1)[0]
        assert_almost_equal(expected, got, decimal=5)

    def test_negxplus1_cuda(self):
        self._negxplus1_cuda(TensorProto.FLOAT)
        self._negxplus1_cuda(TensorProto.FLOAT16)


if __name__ == "__main__":
    unittest.main()
