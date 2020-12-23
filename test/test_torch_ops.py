import unittest
from onnx import load
import torch
import onnxruntime as _ort
import io
import numpy

from torch.onnx import register_custom_op_symbolic
from onnxruntime_customops import (
    onnx_op,
    get_library_path as _get_library_path)


def my_inverse(g, self):
    return g.op("ai.onnx.contrib::Inverse", self)


class CustomInverse(torch.nn.Module):
    def forward(self, x):
        return torch.inverse(x) + x


class TestPyTorchCustomOp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        @onnx_op(op_type="Inverse")
        def inverse(x):
            # the user custom op implementation here:
            return numpy.linalg.inv(x)

    def test_custom_pythonop_pytorch(self):

        # register_custom_op_symbolic(
        #   '<namespace>::inverse', my_inverse, <opset_version>)
        register_custom_op_symbolic('::inverse', my_inverse, 1)

        x = torch.randn(3, 3)

        # Export model to ONNX
        f = io.BytesIO()
        torch.onnx.export(CustomInverse(), (x,), f)
        onnx_model = load(io.BytesIO(f.getvalue()))
        self.assertIn('domain: "ai.onnx.contrib"', str(onnx_model))

        model = CustomInverse()
        pt_outputs = model(x)

        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())

        # Run the exported model with ONNX Runtime
        ort_sess = _ort.InferenceSession(f.getvalue(), so)
        ort_inputs = dict((ort_sess.get_inputs()[i].name, input.cpu().numpy())
                          for i, input in enumerate((x,)))
        ort_outputs = ort_sess.run(None, ort_inputs)

        # Validate PyTorch and ONNX Runtime results
        numpy.testing.assert_allclose(pt_outputs.cpu().numpy(),
                                      ort_outputs[0], rtol=1e-03, atol=1e-05)


if __name__ == "__main__":
    unittest.main()
