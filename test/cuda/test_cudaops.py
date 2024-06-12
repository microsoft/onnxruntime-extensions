import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
from onnx import helper, numpy_helper, onnx_pb as onnx_proto, TensorProto
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_scatternd import _scatter_nd_impl
from onnxruntime_extensions import make_onnx_model
from onnxruntime_extensions import get_library_path as _get_library_path

import onnxruntime as _ort


def has_cuda():
    return "CUDAExecutionProvider" in _ort.get_available_providers()


class ScatterNDOfShape(OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, shape, indices, updates, reduction=None, strategy=None):
        data = np.zeros(shape, dtype=updates.dtype)
        y = _scatter_nd_impl(data, indices, updates, reduction=reduction)
        return (y,)


class NegXPlus1(OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, X):
        return (1 - X,)


class Transpose2DCastFP16(OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, X):
        return (X.T.to(np.float16),)


class Transpose2DCastFP32(OpRun):
    op_domain = "ai.onnx.contrib"

    def _run(self, X):
        return (X.T.to(np.float32),)


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

    def _mulmulsigmoid_cuda(self, itype, broad=False, atol=1e-5, rtol=1e-3):
        model1 = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Mul", ["X", "Y"], ["xy"]),
                    helper.make_node("Sigmoid", ["Y"], ["sy"]),
                    helper.make_node("Mul", ["xy", "sy"], ["final"]),
                ],
                "nd",
                [
                    helper.make_tensor_value_info("X", itype, [None, None, None]),
                    helper.make_tensor_value_info("Y", itype, [None, None, None]),
                ],
                [helper.make_tensor_value_info("final", itype, [None, None, None])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "MulMulSigmoid",
                        ["X", "Y"],
                        ["final"],
                        domain="ai.onnx.contrib",
                    )
                ],
                "nd",
                [
                    helper.make_tensor_value_info("X", itype, [None, None, None]),
                    helper.make_tensor_value_info("Y", itype, [None, None, None]),
                ],
                [helper.make_tensor_value_info("final", itype, [None, None, None])],
            ),
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("ai.onnx.contrib", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        shapex = (1, 2, 3) if broad else (3, 2, 3)
        shapey = (3, 2, 3)
        x = (np.arange(np.prod(shapex)) + 1).reshape(shapex).astype(dtype)
        y = (np.arange(np.prod(shapey)) + 2).reshape(shapey).astype(dtype)
        x /= x.size
        y /= y.size

        feeds1 = dict(X=x, Y=y)
        ref = ReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)[0]

        opts = _ort.SessionOptions()
        opts.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"])
        got = sess.run(None, feeds1)[0]
        assert_allclose(expected, got, atol=atol, rtol=rtol)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_mulmulsigmoid_cuda(self):
        self._mulmulsigmoid_cuda(TensorProto.FLOAT)
        self._mulmulsigmoid_cuda(TensorProto.FLOAT16)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_mulmulsigmoid_cuda_broadcast(self):
        self._mulmulsigmoid_cuda(TensorProto.FLOAT, True)
        self._mulmulsigmoid_cuda(TensorProto.FLOAT16, True)

    def _mul_sigmoid_cuda(self, itype):
        model1 = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Sigmoid", ["X"], ["sx"]),
                    helper.make_node("Mul", ["X", "sx"], ["Y"]),
                ],
                "nd",
                [helper.make_tensor_value_info("X", itype, [None, None, None])],
                [helper.make_tensor_value_info("Y", itype, [None, None, None])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "MulSigmoid",
                        ["X"],
                        ["Y"],
                        domain="ai.onnx.contrib",
                    )
                ],
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
        x = (np.arange(18) + 1).reshape((3, 2, 3)).astype(dtype)

        feeds1 = dict(X=x)
        ref = ReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)[0]

        opts = _ort.SessionOptions()
        opts.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"])
        got = sess.run(None, feeds1)[0]
        assert_allclose(expected, got, atol=1e-5 if itype == TensorProto.FLOAT else 1e-2)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_mul_sigmoid_cuda(self):
        self._mul_sigmoid_cuda(TensorProto.FLOAT)
        self._mul_sigmoid_cuda(TensorProto.FLOAT16)

    def _negxplus1_cuda(self, itype):
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

        opts = _ort.SessionOptions()
        opts.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"])
        got = sess.run(None, feeds1)[0]
        assert_almost_equal(expected, got, decimal=5)

    @unittest.skipIf(not has_cuda(), reason="CUDA is missing")
    def test_cuda_negxplus1(self):
        self._negxplus1_cuda(TensorProto.FLOAT)
        self._negxplus1_cuda(TensorProto.FLOAT16)

    def _addmul_shared_input_cuda(self, itype, op_type, shapea=(3, 2, 3), shapeb=(3, 2, 3), shapec=(3, 2, 3)):
        model1 = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(op_type, ["X", "Y"], ["XY"]),
                    helper.make_node(op_type, ["X", "Z"], ["XZ"]),
                ],
                "nd",
                [
                    helper.make_tensor_value_info("X", itype, [None, None, None]),
                    helper.make_tensor_value_info("Y", itype, [None, None, None]),
                    helper.make_tensor_value_info("Z", itype, [None, None, None]),
                ],
                [
                    helper.make_tensor_value_info("XY", itype, [None, None, None]),
                    helper.make_tensor_value_info("XZ", itype, [None, None, None]),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        f"{op_type}SharedInput",
                        ["X", "Y", "Z"],
                        ["XY", "XZ"],
                        domain="ai.onnx.contrib",
                    )
                ],
                "nd",
                [
                    helper.make_tensor_value_info("X", itype, [None, None, None]),
                    helper.make_tensor_value_info("Y", itype, [None, None, None]),
                    helper.make_tensor_value_info("Z", itype, [None, None, None]),
                ],
                [
                    helper.make_tensor_value_info("XY", itype, [None, None, None]),
                    helper.make_tensor_value_info("XZ", itype, [None, None, None]),
                ],
            ),
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("ai.onnx.contrib", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        x = (np.arange(np.prod(shapea)) + 1).reshape((shapea)).astype(dtype)
        y = (np.arange(np.prod(shapeb)) + 2).reshape((shapeb)).astype(dtype)
        z = (np.arange(np.prod(shapec)) + 3).reshape((shapec)).astype(dtype)

        feeds1 = dict(X=x, Y=y, Z=z)
        ref = ReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)

        opts = _ort.SessionOptions()
        opts.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"])
        got = sess.run(None, feeds1)
        for i in range(2):
            assert_almost_equal(expected[i], got[i])

    @unittest.skipIf(not has_cuda(), reason="CUDA is missing")
    def test_add_shared_input_cuda(self):
        self._addmul_shared_input_cuda(TensorProto.FLOAT, "Add")
        self._addmul_shared_input_cuda(TensorProto.FLOAT16, "Add")

    @unittest.skipIf(not has_cuda(), reason="CUDA is missing")
    def test_mul_shared_input_cuda(self):
        self._addmul_shared_input_cuda(TensorProto.FLOAT, "Mul")
        self._addmul_shared_input_cuda(TensorProto.FLOAT16, "Mul")

    @unittest.skipIf(not has_cuda(), reason="CUDA is missing")
    def test_add_shared_input_cuda_broadcast1(self):
        self._addmul_shared_input_cuda(
            TensorProto.FLOAT,
            "Add",
            shapea=(3, 2, 3),
            shapeb=(1, 2, 3),
            shapec=(1, 2, 3),
        )
        self._addmul_shared_input_cuda(
            TensorProto.FLOAT16,
            "Add",
            shapea=(3, 2, 3),
            shapeb=(1, 2, 3),
            shapec=(1, 2, 3),
        )

    @unittest.skipIf(not has_cuda(), reason="CUDA is missing")
    def test_add_shared_input_cuda_broadcast2(self):
        self._addmul_shared_input_cuda(
            TensorProto.FLOAT,
            "Add",
            shapea=(1, 2, 3),
            shapeb=(3, 2, 3),
            shapec=(3, 2, 3),
        )
        self._addmul_shared_input_cuda(
            TensorProto.FLOAT16,
            "Add",
            shapea=(1, 2, 3),
            shapeb=(3, 2, 3),
            shapec=(3, 2, 3),
        )

    def _scatternd_of_shape_optimize_cuda(self, optimize, dim3, itype):
        indices_shape = ["i", "j", 1] if dim3 else ["j", 1]
        updates_shape = ["i", "j", "b"] if dim3 else ["j", "b"]

        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "ScatterNDOfShape",
                        inputs=["shape", "indices", "updates"],
                        outputs=["y"],
                        reduction="add",
                        strategy="optimize" if optimize else "none",
                        domain="ai.onnx.contrib",
                    )
                ],
                "nd",
                [
                    helper.make_tensor_value_info("shape", TensorProto.INT64, [2]),
                    helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
                    helper.make_tensor_value_info("updates", itype, updates_shape),
                ],
                [helper.make_tensor_value_info("y", itype, [None, None])],
            ),
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("ai.onnx.contrib", 1),
            ],
            ir_version=9,
        )

        if dim3:
            shape = (128, 1024)
            indices = np.zeros((2, 64, 1)).astype(np.int64)
            indices[:, ::2, 0] = 87
            indices[:, ::3, 0] = 85
            updates = np.ones((2, 64, 1024)).astype(np.float32)
        else:
            shape = (128, 1024)
            indices = np.zeros((128, 1)).astype(np.int64)
            indices[::2, 0] = 87
            indices[::3, 0] = 85
            updates = np.ones((128, 1024)).astype(np.float32)
        if itype != 1:
            updates = updates.astype(np.float16)
        feeds = dict(shape=np.array(shape, dtype=np.int64), indices=indices, updates=updates)

        ref = ReferenceEvaluator(model, new_ops=[ScatterNDOfShape])
        expected = ref.run(None, feeds)[0]

        opts = _ort.SessionOptions()
        opts.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(model.SerializeToString(), opts, providers=["CUDAExecutionProvider"])
        ro = None
        got = sess.run(None, feeds, ro)[0]
        self.assertEqual(expected.tolist(), got.tolist())

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_scatternd_of_shape_optimize_cuda(self):
        with self.subTest(optimize=True, dim3=True):
            self._scatternd_of_shape_optimize_cuda(True, True, TensorProto.FLOAT)
        self._scatternd_of_shape_optimize_cuda(False, False, TensorProto.FLOAT)
        self._scatternd_of_shape_optimize_cuda(False, True, TensorProto.FLOAT)
        with self.subTest(optimize=True, dim3=False):
            self._scatternd_of_shape_optimize_cuda(True, False, TensorProto.FLOAT)
        with self.subTest(optimize=True, dim3=True, itype=TensorProto.FLOAT16):
            self._scatternd_of_shape_optimize_cuda(True, True, TensorProto.FLOAT16)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_scatternd_of_shape_standalone_cuda(self):
        self._scatternd_of_shape_cuda("add", 0, TensorProto.FLOAT)
        self._scatternd_of_shape_cuda("add", 0, TensorProto.FLOAT16)
        self._scatternd_of_shape_cuda("add", 1, TensorProto.FLOAT)
        self._scatternd_of_shape_cuda("add", 1, TensorProto.FLOAT16)

    def _masked_scatternd_of_shape_cuda(self, reduction, line, itype, big):
        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16

        model1 = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Equal", ["indices", "mone"], ["masked_indices"]),
                    helper.make_node(
                        "Where",
                        ["masked_indices", "zero", "updates"],
                        ["masked_updates"],
                    ),
                    helper.make_node(
                        "ScatterND",
                        inputs=["data", "indices", "masked_updates"],
                        outputs=["y"],
                        reduction=reduction,
                    ),
                ],
                "nd",
                [
                    helper.make_tensor_value_info("data", itype, [None, None]),
                    helper.make_tensor_value_info("indices", TensorProto.INT64, [None, None, 1]),
                    helper.make_tensor_value_info("updates", itype, [None, None, None]),
                ],
                [helper.make_tensor_value_info("y", itype, [None, None])],
                [
                    numpy_helper.from_array(np.array([-1], dtype=np.int64), name="mone"),
                    numpy_helper.from_array(np.array([0], dtype=dtype), name="zero"),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "MaskedScatterNDOfShape",
                        inputs=["shape", "indices", "updates"],
                        outputs=["y"],
                        reduction=reduction,
                        maskedValue=-1,
                        domain="ai.onnx.contrib",
                    )
                ],
                "nd",
                [
                    helper.make_tensor_value_info("shape", TensorProto.INT64, [None]),
                    helper.make_tensor_value_info("indices", TensorProto.INT64, [None, None, 1]),
                    helper.make_tensor_value_info("updates", itype, [None, None, None]),
                ],
                [helper.make_tensor_value_info("y", itype, [None, None])],
            ),
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("ai.onnx.contrib", 1),
            ],
            ir_version=9,
        )

        if big:
            data = np.zeros((2048, 4096), dtype=dtype)
            indices = np.ones((2, 1024), dtype=np.int64)
            indices = indices[..., np.newaxis]
            shape = tuple(indices.shape[:2]) + (data.shape[-1],)
            updates = (np.arange(np.prod(shape)).reshape(shape) / np.prod(shape)).astype(dtype)
        else:
            data = np.zeros((32, 16), dtype=dtype)
            indices = np.array(
                [
                    [0, 1, 2],
                    [2, 3, 4],
                    [-1, 30, 31],
                    [-1, 7, 8],
                    [10, 11, -1],
                    [20, -1, 21],
                ],
                dtype=np.int64,
            )
            indices = indices[..., np.newaxis]
            shape = (6, 3, data.shape[-1])
            updates = (np.arange(np.prod(shape)).reshape(shape) + 1).astype(dtype)

        feeds1 = dict(data=data, indices=indices, updates=updates)
        feeds2 = dict(shape=np.array(data.shape, dtype=np.int64), indices=indices, updates=updates)
        ref = ReferenceEvaluator(model1)
        expected = ref.run(None, feeds1)[0]

        opts = _ort.SessionOptions()
        opts.register_custom_ops_library(_get_library_path())
        # opts.log_severity_level = 0
        # opts.log_verbosity_level = 0
        sess = _ort.InferenceSession(model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"])
        got = sess.run(None, feeds2)[0]
        assert_almost_equal(expected.tolist(), got.tolist())

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_masked_scatternd_of_shape_standalone_cuda_small(self):
        self._masked_scatternd_of_shape_cuda("add", 0, TensorProto.FLOAT, False)
        self._masked_scatternd_of_shape_cuda("add", 0, TensorProto.FLOAT16, False)
        self._masked_scatternd_of_shape_cuda("add", 1, TensorProto.FLOAT, False)
        self._masked_scatternd_of_shape_cuda("add", 1, TensorProto.FLOAT16, False)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_masked_scatternd_of_shape_standalone_cuda_big(self):
        self._masked_scatternd_of_shape_cuda("add", 0, TensorProto.FLOAT, True)
        self._masked_scatternd_of_shape_cuda("add", 0, TensorProto.FLOAT16, True)
        self._masked_scatternd_of_shape_cuda("add", 1, TensorProto.FLOAT, True)
        self._masked_scatternd_of_shape_cuda("add", 1, TensorProto.FLOAT16, True)

    def _rotary_cuda(self, itype, side, input_shape=(3, 2, 3, 4)):
        model2 = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "Rotary",
                        ["X", "splits"],
                        ["Y"],
                        domain="ai.onnx.contrib",
                        side=side,
                    )
                ],
                "nd",
                [
                    helper.make_tensor_value_info("X", itype, [None, None, None, None]),
                    helper.make_tensor_value_info("splits", TensorProto.INT64, [2]),
                ],
                [helper.make_tensor_value_info("Y", itype, [None, None, None, None])],
            ),
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("ai.onnx.contrib", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        x = (np.arange(np.prod(input_shape)) + 1).reshape(input_shape).astype(dtype)
        splits = np.array([x.shape[-1] // 2, x.shape[-1] // 2], dtype=np.int64)

        expected = x.copy()
        half = x.shape[-1] // 2
        if side == "left":
            expected[:, :, :, :half] = x[:, :, :, half:]
            expected[:, :, :, half:] = -x[:, :, :, :half]
        else:
            expected[:, :, :, :half] = -x[:, :, :, half:]
            expected[:, :, :, half:] = x[:, :, :, :half]

        feeds = dict(X=x, splits=splits)
        opts = _ort.SessionOptions()
        opts.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"])
        got = sess.run(None, feeds)[0]
        assert_almost_equal(expected, got)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_rotary_cuda(self):
        self._rotary_cuda(TensorProto.FLOAT, "left")
        self._rotary_cuda(TensorProto.FLOAT, "right")
        self._rotary_cuda(TensorProto.FLOAT16, "left")
        self._rotary_cuda(TensorProto.FLOAT16, "right")

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_bigger_rotary_cuda(self):
        sh = (2, 2, 1024, 8)
        self._rotary_cuda(TensorProto.FLOAT, "left", input_shape=sh)
        self._rotary_cuda(TensorProto.FLOAT, "right", input_shape=sh)
        self._rotary_cuda(TensorProto.FLOAT16, "left", input_shape=sh)
        self._rotary_cuda(TensorProto.FLOAT16, "right", input_shape=sh)

    def _transpose_cast_cuda(self, itype):
        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        itype2 = TensorProto.FLOAT if itype == TensorProto.FLOAT16 else TensorProto.FLOAT16
        model1 = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node("Transpose", ["X"], ["t"], perm=[1, 0]),
                    helper.make_node("Cast", ["t"], ["Y"], to=itype2),
                ],
                "nd",
                [helper.make_tensor_value_info("X", itype, [None, None])],
                [helper.make_tensor_value_info("Y", itype2, [None, None])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=9,
        )

        model2 = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        ("Transpose2DCastFP16" if itype2 == TensorProto.FLOAT16 else "Transpose2DCastFP32"),
                        ["X"],
                        ["Y"],
                        domain="ai.onnx.contrib",
                    )
                ],
                "nd",
                [helper.make_tensor_value_info("X", itype, [None, None])],
                [helper.make_tensor_value_info("Y", itype2, [None, None])],
            ),
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("ai.onnx.contrib", 1),
            ],
            ir_version=9,
        )

        dtype = np.float32 if itype == TensorProto.FLOAT else np.float16
        x = (np.arange(32 * 32 * 3) + 1).reshape((32, 32 * 3)).astype(dtype)

        feeds1 = dict(X=x)
        ref = ReferenceEvaluator(model1, new_ops=[Transpose2DCastFP16, Transpose2DCastFP32])
        expected = ref.run(None, feeds1)[0]

        opts = _ort.SessionOptions()
        opts.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(model2.SerializeToString(), opts, providers=["CUDAExecutionProvider"])
        got = sess.run(None, feeds1)[0]
        assert_almost_equal(expected, got, decimal=5)

    @unittest.skipIf(not has_cuda(), reason="cuda not available")
    def test_transpose_cast_cuda(self):
        self._transpose_cast_cuda(TensorProto.FLOAT)
        self._transpose_cast_cuda(TensorProto.FLOAT16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
