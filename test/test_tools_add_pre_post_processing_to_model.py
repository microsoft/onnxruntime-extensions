# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import io
import numpy as np
import onnxruntime as ort
import os
import sys
from typing import List

from PIL import Image
from pathlib import Path

# NOTE: This assumes you have created an editable pip install for onnxruntime_extensions by running
# `pip install -e .` from the repo root.
from onnxruntime_extensions import get_library_path
from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
from onnxruntime_extensions.tools import pre_post_processing

script_dir = os.path.dirname(os.path.realpath(__file__))
ort_ext_root = os.path.abspath(os.path.join(script_dir, ".."))
test_data_dir = os.path.join(ort_ext_root, "test", "data", "ppp_vision")


# Function to read the mobilenet labels and adjust for PT vs TF training if needed
# def _get_labels(is_pytorch: bool = True):
#     labels_file = os.path.join(test_data_dir, "TF.ImageNetLabels.txt")
#     labels = []
#     with open(labels_file, 'r') as infile:
#         # skip first 'background' entry if pytorch as that model was not trained with it
#         if is_pytorch:
#             _ = infile.readline()
#
#         for line in infile:
#             labels.append(line.strip())
#
#     assert(len(labels) == 1000 if is_pytorch else 1001)
#     return labels


class TestToolsAddPrePostProcessingToModel(unittest.TestCase):
    def test_pytorch_mobilenet(self):
        input_model = os.path.join(test_data_dir, "pytorch_mobilenet_v2.onnx")
        output_model = os.path.join(test_data_dir, "pytorch_mobilenet_v2.updated.onnx")
        input_image_path = os.path.join(test_data_dir, "wolves.jpg")

        add_ppp.mobilenet(Path(input_model), Path(output_model), add_ppp.ModelSource.PYTORCH)

        def orig_output():
            from torchvision import transforms

            input_image = Image.open(input_image_path)
            preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            input_tensor = preprocess(input_image)
            input_batch = (
                input_tensor.unsqueeze(0).detach().cpu().numpy()
            )  # create a mini-batch as expected by the model

            s = ort.InferenceSession(input_model)
            scores = s.run(None, {"x": np.array(input_batch)})
            scores = np.squeeze(scores)

            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()

            probabilities = softmax(scores)
            return probabilities

        def new_output():
            input_bytes = np.fromfile(input_image_path, dtype=np.uint8)
            so = ort.SessionOptions()
            so.register_custom_ops_library(get_library_path())

            s = ort.InferenceSession(output_model, so)
            probabilities = s.run(None, {"image": np.array(input_bytes)})[0]
            probabilities = np.squeeze(probabilities)  # remove batch dim
            return probabilities

        orig_results = orig_output()
        new_results = new_output()

        orig_idx = np.argmax(orig_results)
        new_idx = np.argmax(new_results)
        self.assertEqual(orig_idx, new_idx)
        # check within 1%. probability values are in range 0..1
        self.assertTrue(abs(orig_results[orig_idx] - new_results[new_idx]) < 0.01)

    def test_tflite_mobilenet(self):
        input_model = os.path.join(test_data_dir, "tflite_mobilenet_v2.onnx")
        output_model = os.path.join(test_data_dir, "tflite_mobilenet_v2.updated.onnx")
        input_image_path = os.path.join(test_data_dir, "wolves.jpg")

        add_ppp.mobilenet(Path(input_model), Path(output_model), add_ppp.ModelSource.TENSORFLOW)

        def orig_output():
            # can still use PT pre-processing as it's using PIL for images.
            # Update the Normalize values to match TF requirements.
            from torchvision import transforms

            input_image = Image.open(input_image_path)
            preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            input_tensor = preprocess(input_image)
            input_batch = (
                input_tensor.unsqueeze(0).detach().cpu().numpy()
            )  # create a mini-batch as expected by the model
            input_batch = np.transpose(input_batch, (0, 2, 3, 1))  # to NHWC format for TF input

            s = ort.InferenceSession(input_model)
            probabilities = s.run(None, {"input": np.array(input_batch)})[0]
            return np.squeeze(probabilities)

        def new_output():
            # TODO: Should we get the ortextensions library path from an env var and if provided run the model?
            input_bytes = np.fromfile(input_image_path, dtype=np.uint8)

            so = ort.SessionOptions()
            so.register_custom_ops_library(get_library_path())

            s = ort.InferenceSession(output_model, so)
            probabilities = s.run(None, {"image": np.array(input_bytes)})[0]
            return np.squeeze(probabilities)  # remove batch dim

        orig_results = orig_output()
        new_results = new_output()

        orig_idx = np.argmax(orig_results)
        new_idx = np.argmax(new_results)
        self.assertEqual(orig_idx, new_idx)
        # check within 1%. probability values are in range 0..1
        self.assertTrue(abs(orig_results[orig_idx] - new_results[new_idx]) < 0.01)

    def test_pytorch_superresolution(self):
        input_model = os.path.join(test_data_dir, "pytorch_super_resolution.onnx")
        output_model = os.path.join(test_data_dir, "pytorch_super_resolution.updated.onnx")
        input_image_path = os.path.join(test_data_dir, "test_superresolution.png")

        # expected output is result of running the model that was manually compared to output from the
        # original pytorch model using torchvision and PIL for pre/post processing. the difference currently is due
        # to ONNX Resize not supporting antialiasing.
        # TODO: When we update to an ORT version with ONNX opset 18 support and enable antialiasing in the Resize
        # (update tools/pre_post_processing/utils.py to set PRE_POST_PROCESSING_ONNX_OPSET to 18) the expected output
        # is in test_superresolution.expected.opset18.png.
        expected_output_image_path = os.path.join(test_data_dir, "test_superresolution.expected.png")

        add_ppp.superresolution(Path(input_model), Path(output_model), "png")

        input_bytes = np.fromfile(input_image_path, dtype=np.uint8)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        s = ort.InferenceSession(output_model, so)

        result_bytes = s.run(None, {"image": np.array(input_bytes)})[0]

        # convert from png to RGB to remove any png encoding diffs
        result_img = Image.open(io.BytesIO(result_bytes))
        result = np.array(result_img.convert("RGB"))
        expected = np.array(Image.open(expected_output_image_path).convert("RGB"))

        # check all pixel values are close. allowing for 0.1% of pixels to differ by 2 at the most.
        #
        # we expect some variance from the floating point operations involved during Resize and conversion of the
        # original image to/from YCbCr. the different instructions used on different hardware can cause diffs, such as
        # whether avx512 is used or not. MacOS seems to be slightly worse though (max of 2)
        diffs = np.absolute(expected.astype(np.int32) - result.astype(np.int32))
        total = np.sum(diffs)
        print(f"Max diff:{diffs.max()} Total diffs:{total}")
        self.assertTrue(diffs.max() < 3 and total < (result.size / 1000))

    def build_testModel_and_get_ref_output_for_tokenizer(
        self, tokenizer_type: str, output_model: Path, pre_steps: List[pre_post_processing.steps.Step] = []
    ):
        # import transformers
        import onnx

        create_named_value = pre_post_processing.utils.create_named_value
        SentencePieceTokenizer = pre_post_processing.steps.SentencePieceTokenizer
        TokenizerParam = pre_post_processing.steps.TokenizerParam
        BertTokenizer = pre_post_processing.steps.BertTokenizer
        tokenizer_step = BertTokenizer
        input_text = ("This is a test sentence",)
        inputs = [create_named_value("inputs", onnx.TensorProto.STRING, ["sentence_length"])]

        if tokenizer_type == "sentencePiece":
            ref_output = [np.array([[0, 3293, 83, 10, 3034, 149357, 2]]), np.array([[1, 1, 1, 1, 1, 1, 1]])]
            # tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
            tokenizer_args = TokenizerParam(
                vocab_or_file=os.path.join(test_data_dir, "../sentencepiece.bpe.model"),
                bos_token_id=0,
            )
            tokenizer_step = SentencePieceTokenizer
        elif tokenizer_type == "BertTokenizer":
            ref_output = [
                np.array([[2, 236, 118, 16, 1566, 875, 643, 3]]),
                np.array([[0, 0, 0, 0, 0, 0, 0, 0]]),
                np.array([[1, 1, 1, 1, 1, 1, 1, 1]]),
            ]
            # tokenizer = transformers.AutoTokenizer.from_pretrained("lordtt13/emo-mobilebert")
            tokenizer_args = TokenizerParam(vocab_or_file=os.path.join(test_data_dir, "../bert.vocab"), do_lower_case=True)
        elif tokenizer_type in ["hfBertTokenizer", "hfBertTokenizer_with_decoder"]:
            ref_output = (
                [
                    np.array([[2, 236, 118, 16, 1566, 875, 643, 3, 236, 118, 978, 1566, 875, 643, 3]]),
                    np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]),
                    np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
                ]
                if tokenizer_type == "hfBertTokenizer"
                else [np.array(["[CLS]"])]
            )
            # tokenizer = transformers.AutoTokenizer.from_pretrained("lordtt13/emo-mobilebert")
            tokenizer_args = TokenizerParam(vocab_or_file=os.path.join(test_data_dir, "../hfbert.vocab"), do_lower_case=True)
            input_text = ("This is a test sentence", "This is another test sentence")
            inputs = [create_named_value("inputs", onnx.TensorProto.STRING, [2, "sentence_length"])]
        else:
            raise Exception("Unknown tokenizer")

        pipeline = pre_post_processing.PrePostProcessor(inputs)
        # ref_output = list(tokenizer(*input_text, return_tensors="np").values())

        pipeline.add_pre_processing([tokenizer_step(tokenizer_args)] + pre_steps)
        if tokenizer_type == "hfBertTokenizer_with_decoder":
            pipeline.add_post_processing([pre_post_processing.steps.BertTokenizerQATaskDecoder(tokenizer_args)])
            input_model = onnx.load(os.path.join(test_data_dir, "../bert_qa_decoder_base.onnx"))
        else:
            input_model = onnx.ModelProto()
            input_model.opset_import.extend([onnx.helper.make_operatorsetid("", 16)])
        new_model = pipeline.run(input_model)
        onnx.save_model(new_model, output_model)
        return input_text, ref_output

    def test_sentencePiece_tokenizer(self):
        output_model = (Path(test_data_dir) / "../sentencePiece.onnx").resolve()

        input_text, ref_output = self.build_testModel_and_get_ref_output_for_tokenizer("sentencePiece", output_model)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        s = ort.InferenceSession(str(output_model), so, providers=["CPUExecutionProvider"])

        result = s.run(None, {s.get_inputs()[0].name: np.array([*input_text])})

        # SentencePieceTokenizer in ORT is round to zero, so we need to use atol=1
        self.assertEqual(np.allclose(result[0], ref_output[0], atol=1), True)
        self.assertEqual(np.allclose(result[1], ref_output[1]), True)

    def test_bert_tokenizer(self):
        output_model = (Path(test_data_dir) / "../bert_tokenizer.onnx").resolve()
        input_text, ref_output = self.build_testModel_and_get_ref_output_for_tokenizer("BertTokenizer", output_model)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        s = ort.InferenceSession(str(output_model), so, providers=["CPUExecutionProvider"])

        result = s.run(None, {s.get_inputs()[0].name: np.array([*input_text])})

        self.assertEqual(np.allclose(result[0], ref_output[0]), True)
        self.assertEqual(np.allclose(result[1], ref_output[2]), True)
        self.assertEqual(np.allclose(result[2], ref_output[1]), True)

    def test_hfBert_tokenizer(self):
        output_model = (Path(test_data_dir) / "../hfbert_tokenizer.onnx").resolve()
        input_text, ref_output = self.build_testModel_and_get_ref_output_for_tokenizer("hfBertTokenizer", output_model)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        s = ort.InferenceSession(str(output_model), so, providers=["CPUExecutionProvider"])

        result = s.run(None, {s.get_inputs()[0].name: np.array([[input_text[0]], [input_text[1]]])})

        self.assertEqual(np.allclose(result[0], ref_output[0]), True)
        self.assertEqual(np.allclose(result[1], ref_output[2]), True)
        self.assertEqual(np.allclose(result[2], ref_output[1]), True)

    def test_qatask_with_tokenizer(self):
        output_model = (Path(test_data_dir) / "../hfbert_tokenizer.onnx").resolve()
        input_text, ref_output = self.build_testModel_and_get_ref_output_for_tokenizer(
            "hfBertTokenizer_with_decoder", output_model, [pre_post_processing.steps.BertTokenizerQATask()]
        )

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        s = ort.InferenceSession(str(output_model), so, providers=["CPUExecutionProvider"])

        result = s.run(None, {s.get_inputs()[0].name: np.array([[input_text[0]], [input_text[1]]])})

        self.assertEqual(result[0][0], ref_output[0][0])

    # Corner Case
    def test_debug_step(self):
        from onnxruntime_extensions.tools import pre_post_processing
        import onnx

        create_named_value = pre_post_processing.utils.create_named_value

        # multiple DebugSteps are stringed together
        input_model = os.path.join(test_data_dir, "pytorch_super_resolution.onnx")
        inputs = [create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]
        pipeline = pre_post_processing.PrePostProcessor(inputs)
        pipeline.add_post_processing(
            [pre_post_processing.Debug(), pre_post_processing.Debug(3), pre_post_processing.Debug(4)]
        )
        new_model = pipeline.run(onnx.load(input_model))
        self.assertEqual(len(new_model.graph.output), 4)


if __name__ == "__main__":
    unittest.main()
