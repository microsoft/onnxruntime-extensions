import io
import onnx
import numpy
import unittest
import platform
import torch
import torchvision
import onnxruntime as _ort

from onnx import load
from torch.onnx import register_custom_op_symbolic
from onnxruntime_extensions import (
    PyOp,
    onnx_op,
    hook_model_op,
    get_library_path as _get_library_path)

from onnxruntime_extensions.eager_op import EagerOp


def my_inverse(g, self):
    return g.op("ai.onnx.contrib::Inverse", self)


register_custom_op_symbolic('::inverse', my_inverse, 1)


def my_all(g, self):
    return g.op("ai.onnx.contrib::All", self)


register_custom_op_symbolic('::all', my_all, 1)


class CustomInverse(torch.nn.Module):
    def forward(self, x, y):
        ress = torch.inverse(x) + x
        return ress, torch.all(y)


class TestPyTorchCustomOp(unittest.TestCase):

    _hooked = False

    @classmethod
    def setUpClass(cls):

        @onnx_op(op_type="Inverse")
        def inverse(x):
            # the user custom op implementation here:
            return numpy.linalg.inv(x)

        @onnx_op(op_type='All', inputs=[PyOp.dt_bool], outputs=[PyOp.dt_bool])
        def op_all(x):
            return numpy.all(x)

    def test_custom_pythonop_pytorch(self):

        # register_custom_op_symbolic(
        #   '<namespace>::inverse', my_inverse, <opset_version>)

        x0, x1 = torch.randn(3, 3), torch.tensor([True, False])

        # Export model to ONNX
        f = io.BytesIO()
        torch.onnx.export(CustomInverse(), (x0, x1), f, opset_version=12)
        onnx_model = load(io.BytesIO(f.getvalue()))
        self.assertIn('domain: "ai.onnx.contrib"', str(onnx_model))

        model = CustomInverse()
        onnx.save_model(onnx_model, 'temp_pytorchcustomop.onnx')
        pt_outputs = model(x0, x1)

        run_ort = EagerOp.from_model(onnx_model)
        ort_outputs = run_ort(x0.numpy(), x1.numpy())

        # Validate PyTorch and ONNX Runtime results
        numpy.testing.assert_allclose(pt_outputs[0].numpy(),
                                      ort_outputs[0], rtol=1e-03, atol=1e-05)

    @staticmethod
    def on_hook(*x):
        TestPyTorchCustomOp._hooked = True
        return x

    @unittest.skipIf(platform.system() == 'Darwin', "pytorch.onnx crashed for this case!")
    def test_pyop_hooking(self):    # type: () -> None
        model = torchvision.models.mobilenet_v2(pretrained=False)
        x = torch.rand(1, 3, 224, 224)
        with io.BytesIO() as f:
            torch.onnx.export(model, (x, ), f)
            model = onnx.load_model_from_string(f.getvalue())

            self.assertTrue(model.graph.node[5].op_type == 'Conv')
            hkd_model = hook_model_op(model, model.graph.node[5].name, TestPyTorchCustomOp.on_hook, [PyOp.dt_float] * 3)

            so = _ort.SessionOptions()
            so.register_custom_ops_library(_get_library_path())
            sess = _ort.InferenceSession(hkd_model.SerializeToString(), so)
            TestPyTorchCustomOp._hooked = False
            sess.run(None, {'input.1': x.numpy()})
            self.assertTrue(TestPyTorchCustomOp._hooked)


if __name__ == "__main__":
    unittest.main()
