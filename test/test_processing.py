import onnx
import numpy
import torch
import unittest
from typing import List
from PIL import Image
from distutils.version import LooseVersion
from onnxruntime_extensions import OrtPyFunction
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


@torch.jit.script
def _broadcasting_add(input_list: List[torch.Tensor]) -> torch.Tensor:
    return input_list[1] + input_list[0]


class _SequenceTensorModel(pnp.ProcessingScriptModule):
    def forward(self, img_list: List[torch.Tensor]) -> torch.Tensor:
        return _broadcasting_add(img_list)


class _MobileNetProcessingModule(pnp.ProcessingScriptModule):
    def __init__(self, oxml):
        super(_MobileNetProcessingModule, self).__init__()
        self.model_function_id = pnp.create_model_function(oxml)
        self.pre_proc = torch.jit.trace(pnp.PreMobileNet(224), torch.zeros(224, 224, 3, dtype=torch.float32))
        self.post_proc = pnp.ImageNetPostProcessing()

    def forward(self, img):
        proc_input = self.pre_proc(img)
        return self.post_proc.forward(pnp.invoke_onnx_model(self.model_function_id, proc_input))


@unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.9"), 'Only tested the latest PyTorch')
class TestPreprocessing(unittest.TestCase):
    def test_imagenet_preprocessing(self):
        mnv2 = onnx.load_model(get_test_data_file('data', 'mobilev2.onnx'))

        # load an image
        img = Image.open(get_test_data_file('data', 'pineapple.jpg'))
        img = torch.from_numpy(numpy.asarray(img.convert('RGB')))

        full_models = pnp.SequenceProcessingModule(
            pnp.SequenceProcessingModule(
                pnp.PreMobileNet(224),
                mnv2), pnp.PostMobileNet())

        ids, probabilities = full_models.forward(img)
        full_model_func = OrtPyFunction.from_model(
            pnp.export(full_models, img, opset_version=11, output_path='temp_imagenet.onnx'))
        actual_ids, actual_result = full_model_func(img)
        numpy.testing.assert_allclose(probabilities.numpy(), actual_result, rtol=1e-3)
        self.assertEqual(ids[0, 0].item(), 953)  # 953 is pineapple class id in the imagenet dataset

    @unittest.skip
    def test_gpt2_preprocessing(self):
        cfg = GPT2Config(n_layer=3)
        gpt2_m = _GPT2LMHeadModel(cfg)
        gpt2_m.eval().to('cpu')
        full_model = pnp.SequenceProcessingModule(
            pnp.PreHuggingFaceGPT2(vocab_file=get_test_data_file('data', 'gpt2.vocab'),
                                   merges_file=get_test_data_file('data', 'gpt2.merges.txt')),
            gpt2_m)
        test_sentence = ["Test a sentence"]
        expected = full_model.forward(test_sentence)
        model = pnp.export(full_model, test_sentence, opset_version=12, do_constant_folding=False)
        mfunc = OrtPyFunction.from_model(model)
        actuals = mfunc(test_sentence)
        # the random weight may generate a large diff in result, test the shape only.
        self.assertTrue(numpy.allclose(expected.size(), actuals.shape))

    @unittest.skip
    def test_sequence_tensor(self):
        seq_m = _SequenceTensorModel()
        test_input = [torch.from_numpy(_i) for _i in [
            numpy.array([1]), numpy.array([3, 4]), numpy.array([5, 6])]]
        res = seq_m.forward(test_input)
        numpy.testing.assert_allclose(res, numpy.array([4, 5]))
        if LooseVersion(torch.__version__) >= LooseVersion("1.11"):
            # The ONNX exporter fixing for sequence tensor support only released in 1.11 and the above.
            oxml = pnp.export(seq_m,
                              test_input,
                              oppset_version=12,
                              output_file='temp_seqtest.onnx')
            # TODO: ORT doesn't accept the default empty element type of a sequence type.
            oxml.graph.input[0].type.sequence_type.elem_type.CopyFrom(
                onnx.helper.make_tensor_type_proto(onnx.onnx_pb.TensorProto.INT32, []))
            mfunc = OrtPyFunction.from_model(oxml)
            o_res = mfunc(test_input)
            numpy.testing.assert_allclose(res, o_res)

    @unittest.skip
    def test_functional_processing(self):
        # load an image
        img = Image.open(get_test_data_file('data', 'pineapple.jpg')).convert('RGB')
        img = torch.from_numpy(numpy.asarray(img))

        pipeline = _MobileNetProcessingModule(onnx.load_model(get_test_data_file('data', 'mobilev2.onnx')))
        ids, probabilities = pipeline.forward(img)

        full_model_func = OrtPyFunction.from_model(
            pnp.export(pipeline, img, opset_version=11, output_path='temp_func.onnx'))
        actual_ids, actual_result = full_model_func(img)
        numpy.testing.assert_allclose(probabilities.numpy(), actual_result, rtol=1e-3)
        self.assertEqual(ids[0, 0].item(), 953)  # 953 is pineapple class id in the imagenet dataset


if __name__ == "__main__":
    unittest.main()
