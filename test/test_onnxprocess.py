import io
import onnx
import unittest
import torchvision
import numpy as np
from onnxruntime_extensions import eager_op, hook_model_op, PyOp
from onnxruntime_extensions.onnxprocess import torch_wrapper as torch
from onnxruntime_extensions.onnxprocess import trace_for_onnx, pyfunc_from_model


class TestTorchE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        cls.argmax_input = None

    @staticmethod
    def on_hook(*x):
        TestTorchE2E.argmax_input = x[0]
        return x

    def test_range(self):
        num = 10

        f = io.BytesIO()
        with trace_for_onnx(num, names=['count']) as tc_sess:
            num_in = tc_sess.get_inputs()[0]
            done = torch.tensor(True)
            st_0 = torch.tensor(0)
            cfg = torch.control_flow()
            for _ in cfg.loop(num_in, done, st_0):
                iter_num, *v = _
                cfg.flow_output(done, st_0, iter_num + 0)

            *_, rout = cfg.finalize()
            tc_sess.save_as_onnx(f, rout)

        m = onnx.load_model_from_string(f.getvalue())
        onnx.save_model(m, 'temp_range.onnx')
        fu_m = eager_op.EagerOp.from_model(m)
        result = fu_m(num)
        np.testing.assert_array_equal(result, np.array(range(num)))

    def test_sequence(self):
        input_text = ['test sentence', 'sentence 2']
        f = io.BytesIO()
        with trace_for_onnx(input_text, names=['in_text']) as tc_sess:
            tc_inputs = tc_sess.get_inputs()[0]
            batchsize = tc_inputs.size()[0]
            shape = [batchsize, 2]
            fuse_output = torch.zeros(*shape).size()
            tc_sess.save_as_onnx(f, fuse_output)

        m = onnx.load_model_from_string(f.getvalue())
        onnx.save_model(m, 'temp_test00.onnx')
        fu_m = eager_op.EagerOp.from_model(m)
        result = fu_m(input_text)
        np.testing.assert_array_equal(result, [2, 2])

    def test_imagenet_postprocess(self):
        mb_core_path = 'temp_mobilev2.onnx'
        mb_full_path = 'temp_mobilev2_full.onnx'
        dummy_input = torch.randn(10, 3, 224, 224)
        np_input = dummy_input.numpy()
        torch.onnx.export(self.mobilenet, dummy_input, mb_core_path, opset_version=11)
        mbnet2 = pyfunc_from_model(mb_core_path)

        with trace_for_onnx(dummy_input, names=['b10_input']) as tc_sess:
            scores = mbnet2(*tc_sess.get_inputs())
            probabilities = torch.softmax(scores, dim=1)
            batch_top1 = probabilities.argmax(dim=1)

            np_argmax = probabilities.numpy()  # for the result comparison
            np_output = batch_top1.numpy()

            tc_sess.save_as_onnx(mb_full_path, batch_top1)

        hkdmdl = hook_model_op(onnx.load_model(mb_full_path), 'argmax', self.on_hook, [PyOp.dt_float])
        mbnet2_full = eager_op.EagerOp.from_model(hkdmdl)
        batch_top1_2 = mbnet2_full(np_input)
        np.testing.assert_allclose(np_argmax, self.argmax_input, rtol=1e-5)
        np.testing.assert_array_equal(batch_top1_2, np_output)


if __name__ == "__main__":
    unittest.main()
