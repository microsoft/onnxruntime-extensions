import onnx
import numpy
import torch
import unittest
from typing import List, Tuple
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
        self.post_proc = torch.jit.trace(pnp.ImageNetPostProcessing(), torch.zeros(1, 1000, dtype=torch.float32))

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proc_input = self.pre_proc(img)
        return self.post_proc.forward(pnp.invoke_onnx_model1(self.model_function_id, proc_input))


@unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.9"), 'Not works with older PyTorch')
class TestPreprocessing(unittest.TestCase):
    def test_imagenet_preprocessing(self):
        mnv2 = onnx.load_model(get_test_data_file('data', 'mobilev2.onnx'))

        # load an image
        img = Image.open(get_test_data_file('data', 'pineapple.jpg'))
        img = torch.from_numpy(numpy.asarray(img.convert('RGB')))

        full_models = pnp.SequentialProcessingModule(pnp.PreMobileNet(224),
                                                     mnv2,
                                                     pnp.PostMobileNet())
        ids, probabilities = full_models.forward(img)
        name_i = 'image'
        full_model_func = OrtPyFunction.from_model(
            pnp.export(full_models,
                       img,
                       opset_version=11,
                       output_path='temp_imagenet.onnx',
                       input_names=[name_i],
                       dynamic_axes={name_i: [0, 1]}))
        actual_ids, actual_result = full_model_func(img.numpy())
        numpy.testing.assert_allclose(probabilities.numpy(), actual_result, rtol=1e-3)
        self.assertEqual(ids[0, 0].item(), 953)  # 953 is pineapple class id in the imagenet dataset

    def test_gpt2_preprocessing(self):
        cfg = GPT2Config(n_layer=3)
        gpt2_m = _GPT2LMHeadModel(cfg)
        gpt2_m.eval().to('cpu')

        test_sentence = ["Test a sentence"]
        tok = pnp.PreHuggingFaceGPT2(vocab_file=get_test_data_file('data', 'gpt2.vocab'),
                                     merges_file=get_test_data_file('data', 'gpt2.merges.txt'))
        inputs = tok.forward(test_sentence)
        pnp.export(tok, test_sentence, opset_version=12, output_path='temp_tok2.onnx')

        with open('temp_gpt2lmh.onnx', 'wb') as f:
            torch.onnx.export(gpt2_m, inputs, f, opset_version=12, do_constant_folding=False)
        pnp.export(gpt2_m, *inputs, opset_version=12, do_constant_folding=False)
        full_model = pnp.SequentialProcessingModule(tok, gpt2_m)
        expected = full_model.forward(test_sentence)
        model = pnp.export(full_model, test_sentence, opset_version=12, do_constant_folding=False)
        mfunc = OrtPyFunction.from_model(model)
        actuals = mfunc(test_sentence)
        # the random weight may generate a large diff in result, test the shape only.
        self.assertTrue(numpy.allclose(expected.size(), actuals.shape))

    def test_sequence_tensor(self):
        seq_m = _SequenceTensorModel()
        test_input = [torch.from_numpy(_i) for _i in [
            numpy.array([1]).astype(numpy.int64),
            numpy.array([3, 4]).astype(numpy.int64),
            numpy.array([5, 6]).astype(numpy.int64)]]
        res = seq_m.forward(test_input)
        numpy.testing.assert_allclose(res, numpy.array([4, 5]))
        if LooseVersion(torch.__version__) >= LooseVersion("1.11"):
            # The fixing for the sequence tensor support is only released in 1.11 and the above.
            oxml = pnp.export(seq_m,
                              [test_input],
                              opset_version=12,
                              output_path='temp_seqtest.onnx')
            # TODO: ORT doesn't accept the default empty element type of a sequence type.
            oxml.graph.input[0].type.sequence_type.elem_type.CopyFrom(
                onnx.helper.make_tensor_type_proto(onnx.onnx_pb.TensorProto.INT64, []))
            mfunc = OrtPyFunction.from_model(oxml)
            o_res = mfunc([_i.numpy() for _i in test_input])
            numpy.testing.assert_allclose(res, o_res)

    @unittest.skipIf(LooseVersion(torch.__version__) < LooseVersion("1.11"),
                     'PythonOp bug fixing on Pytorch 1.11')
    def test_functional_processing(self):
        # load an image
        img = Image.open(get_test_data_file('data', 'pineapple.jpg')).convert('RGB')
        img = torch.from_numpy(numpy.asarray(img))

        pipeline = _MobileNetProcessingModule(onnx.load_model(get_test_data_file('data', 'mobilev2.onnx')))
        ids, probabilities = pipeline.forward(img)

        full_model_func = OrtPyFunction.from_model(
            pnp.export(pipeline, img, opset_version=11, output_path='temp_func.onnx'))
        actual_ids, actual_result = full_model_func(img.numpy())
        numpy.testing.assert_allclose(probabilities.numpy(), actual_result, rtol=1e-3)
        self.assertEqual(ids[0, 0].item(), 953)  # 953 is pineapple class id in the imagenet dataset


if __name__ == "__main__":
    unittest.main()
