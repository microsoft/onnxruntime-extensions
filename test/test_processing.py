import onnx
import numpy
import torch
import unittest
from typing import List
from PIL import Image
from distutils.version import LooseVersion
from onnxruntime_extensions import PyOrtFunction, ONNXCompose
from onnxruntime_extensions import pnp, get_test_data_file
from transformers import GPT2Config, GPT2LMHeadModel


class _GPT2LMHeadModel(GPT2LMHeadModel):
    """ Here we wrap a class for Onnx model conversion for GPT2LMHeadModel with past state.
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask):
        result = super(_GPT2LMHeadModel, self).forward(input_ids,
                                                       attention_mask=attention_mask,
                                                       return_dict=False)
        # drop the past states
        return result[0]


class _SequenceTensorModel(pnp.ProcessingScriptModule):
    def forward(self, img_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return img_list[0], img_list[1]


class _AddModel(torch.nn.Module):
    def forward(self, input_list: List[torch.Tensor]) -> torch.Tensor:
        return input_list[1] + input_list[0]  # test broadcasting.


@unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.9"), 'Only tested the lastest PyTorch')
class TestPreprocessing(unittest.TestCase):
    def test_imagenet_preprocessing(self):
        mnv2 = onnx.load_model(get_test_data_file(__file__, 'data', 'mobilev2.onnx'))

        # load an image
        img = Image.open(get_test_data_file(__file__, 'data', 'pineapple.jpg'))
        img = numpy.asarray(img.convert('RGB'))

        full_models = ONNXCompose(
            mnv2,
            preprocessors=pnp.PreMobileNet(224),
            postprocessors=pnp.PostMobileNet()
        )

        ids, probabilities = full_models.predict(torch.from_numpy(img))
        full_model_func = PyOrtFunction.from_model(full_models.export(opset_version=11))
        actual_ids, actual_result = full_model_func(img)
        numpy.testing.assert_allclose(probabilities.numpy(), actual_result, rtol=1e-3)
        self.assertEqual(ids[0, 0].item(), 953)  # 953 is pineapple class id in the imagenet dataset

    def test_gpt2_preprocessing(self):
        cfg = GPT2Config(n_layer=3)
        gpt2_m = _GPT2LMHeadModel(cfg)
        gpt2_m.eval().to('cpu')
        full_model = ONNXCompose(
            gpt2_m,
            preprocessors=pnp.PreHuggingFaceGPT2(vocab_file=get_test_data_file(__file__, 'data', 'gpt2.vocab'),
                                                 merges_file=get_test_data_file(__file__, 'data', 'gpt2.merges.txt')))
        test_sentence = ["Test a sentence"]
        expected = full_model.predict(test_sentence)
        model = full_model.export(opset_version=12, do_constant_folding=False)
        mfunc = PyOrtFunction.from_model(model)
        actuals = mfunc(test_sentence)
        # the random weight may generate a large diff in result, test the shape only.
        self.assertTrue(numpy.allclose(expected.size(), actuals.shape))

    def test_sequence_tensor(self):
        seq_m = ONNXCompose(torch.jit.script(_AddModel()), _SequenceTensorModel(), None)
        test_input = [numpy.array([1]), numpy.array([3, 4]), numpy.array([5, 6])]
        res = seq_m.predict(test_input)
        numpy.testing.assert_allclose(res, numpy.array([4, 5]))
        if LooseVersion(torch.__version__) >= LooseVersion("1.11"):
            # The ONNX exporter fixing for sequence tensor only released in 1.11 and the above.
            oxml = seq_m.export(12, output_file='temp_seqtest.onnx')
            # TODO: ORT doesn't accept the default empty element type of a sequence type.
            oxml.graph.input[0].type.sequence_type.elem_type.CopyFrom(
                onnx.helper.make_tensor_type_proto(onnx.onnx_pb.TensorProto.INT32, []))
            mfunc = PyOrtFunction.from_model(oxml)
            o_res = mfunc(test_input)
            numpy.testing.assert_allclose(res, o_res)


if __name__ == "__main__":
    unittest.main()
