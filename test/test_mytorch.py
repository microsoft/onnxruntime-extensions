import onnx
import torch
import unittest
import torchvision
import numpy as np
from onnxruntime_customops import mytorch, eager_op, hook_model_op, PyOp


class TestTorchE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        cls.argmax_input = None

    @staticmethod
    def on_hook(*x):
        TestTorchE2E.argmax_input = x[0]
        return x

    def test_imagenet_postprocess(self):
        mb_core_path = "mobilev2.onnx"
        mb_full_path = "mobilev2_full.onnx"
        dummy_input = torch.randn(10, 3, 224, 224)
        np_input = dummy_input.numpy()
        torch.onnx.export(self.mobilenet, dummy_input, mb_core_path, opset_version=11)

        inputs = mytorch.start_trace(dummy_input, names=['b10_input'])
        mbnet2 = mytorch.op_from_model(mb_core_path)
        scores = mbnet2(*inputs)
        probabilities = mytorch.softmax(scores[0], dim=1)
        batch_top1 = probabilities.argmax(dim=1)
        np_argmax = probabilities.numpy()
        np_output = batch_top1.numpy()

        with mytorch.stop_trace(batch_top1) as tc_sess:
            tc_sess.save_as_onnx(mb_full_path)

        hkdmdl = hook_model_op(onnx.load_model(mb_full_path), 'argmax', self.on_hook, [PyOp.dt_float])
        mbnet2_full = eager_op.EagerOp.from_model(hkdmdl)
        batch_top1_2 = mbnet2_full(np_input)
        np.testing.assert_allclose(np_argmax, self.argmax_input, rtol=1e-5)
        np.testing.assert_array_equal(batch_top1_2, np_output)


if __name__ == "__main__":
    unittest.main()
