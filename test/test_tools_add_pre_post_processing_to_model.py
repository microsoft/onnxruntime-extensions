# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import io
import numpy as np
import onnxruntime as ort
import os
from typing import List

from PIL import Image
from pathlib import Path
from distutils.version import LooseVersion
# NOTE: This assumes you have created an editable pip install for onnxruntime_extensions by running
# `pip install -e .` from the repo root.
from onnxruntime_extensions import get_library_path
from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
from onnxruntime_extensions.tools import pre_post_processing as pre_post_processing
from onnxruntime_extensions.tools.pre_post_processing.steps import *


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

@unittest.skipIf(LooseVersion(ort.__version__) < LooseVersion("1.13"), "only supported in ort 1.13 and above")
class TestToolsAddPrePostProcessingToModel(unittest.TestCase):
    def test_pytorch_mobilenet(self):
        input_model = os.path.join(test_data_dir, "pytorch_mobilenet_v2.onnx")
        output_model = os.path.join(test_data_dir, "pytorch_mobilenet_v2.updated.onnx")
        input_image_path = os.path.join(test_data_dir, "wolves.jpg")

        add_ppp.mobilenet(Path(input_model), Path(output_model), add_ppp.ModelSource.PYTORCH)

        def orig_output():
            from torchvision import transforms
            input_image = Image.open(input_image_path)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(
                0).detach().cpu().numpy()  # create a mini-batch as expected by the model

            s = ort.InferenceSession(input_model, providers=['CPUExecutionProvider'])
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

            s = ort.InferenceSession(output_model, so, providers=['CPUExecutionProvider'])
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
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            input_tensor = preprocess(input_image)
            # create a mini-batch as expected by the model
            input_batch = (input_tensor.unsqueeze(0).detach().cpu().numpy())
            # to NHWC format for TF input
            input_batch = np.transpose(input_batch, (0, 2, 3, 1))

            s = ort.InferenceSession(input_model, providers=['CPUExecutionProvider'])
            probabilities = s.run(None, {"input": np.array(input_batch)})[0]
            return np.squeeze(probabilities)

        def new_output():
            # TODO: Should we get the ortextensions library path from an env var and if provided run the model?
            input_bytes = np.fromfile(input_image_path, dtype=np.uint8)

            so = ort.SessionOptions()
            so.register_custom_ops_library(get_library_path())

            s = ort.InferenceSession(output_model, so, providers=['CPUExecutionProvider'])
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
        from packaging import version
        if version.parse(ort.__version__) >= version.parse("1.14.0"):
            onnx_opset = 18
            expected_output_image_path = os.path.join(test_data_dir, "test_superresolution.expected.opset18.png")
        else:
            onnx_opset = 16
            expected_output_image_path = os.path.join(test_data_dir, "test_superresolution.expected.png")

        add_ppp.superresolution(Path(input_model), Path(output_model), "png", onnx_opset)

        input_bytes = np.fromfile(input_image_path, dtype=np.uint8)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        s = ort.InferenceSession(output_model, so, providers=['CPUExecutionProvider'])

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

    def create_pipeline_and_run_for_tokenizer(self, tokenizer_impl, tokenizer_type,
                                              tokenizer_parameters, output_model: Path):
        import onnx
        create_named_value = pre_post_processing.utils.create_named_value

        inputs = [create_named_value("input_text", onnx.TensorProto.STRING, [1, "num_sentences"])]

        pipeline = pre_post_processing.PrePostProcessor(inputs)
        # ref_output = list(tokenizer(*input_text, return_tensors="np").values())

        pipeline.add_pre_processing([tokenizer_impl])
        if tokenizer_type == "HfBertTokenizer_with_decoder":
            pipeline.add_post_processing([
                (BertTokenizerQADecoder(tokenizer_parameters), [
                 pre_post_processing.utils.IoMapEntry("BertTokenizer", producer_idx=0, consumer_idx=2)])
            ])
            input_model = onnx.load(os.path.join(test_data_dir, "../bert_qa_decoder_base.onnx"))
        else:
            input_model = onnx.ModelProto()
            input_model.opset_import.extend([onnx.helper.make_operatorsetid("", 16)])
        new_model = pipeline.run(input_model)
        onnx.save_model(new_model, output_model)
        return

    def test_sentencepiece_tokenizer(self):
        output_model = (Path(test_data_dir) / "../sentencePiece.onnx").resolve()

        input_text = ("This is a test sentence",)
        ref_output = [np.array([[0, 3293, 83, 10, 3034, 149357, 2]]),
                      np.array([[1, 1, 1, 1, 1, 1, 1]])]
        # tokenizer = transformers.AutoTokenizer.from_pretrained("xlm-roberta-base")
        tokenizer_parameters = TokenizerParam(
            vocab_or_file=os.path.join(test_data_dir, "../sentencepiece.bpe.model"),
            tweaked_bos_id=0,
        )
        tokenizer_impl = SentencePieceTokenizer(tokenizer_parameters, add_eos=True, add_bos=True)
        self.create_pipeline_and_run_for_tokenizer(
            tokenizer_impl, "SentecePieceTokenizer", tokenizer_parameters, output_model)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        s = ort.InferenceSession(str(output_model), so, providers=["CPUExecutionProvider"])

        result = s.run(None, {s.get_inputs()[0].name: np.array([[*input_text]])})

        # SentencePieceTokenizer in ORT is round to zero, so we need to use atol=1
        self.assertEqual(np.allclose(result[0], ref_output[0], atol=1), True)
        self.assertEqual(np.allclose(result[1], ref_output[1]), True)

    def test_bert_tokenizer(self):
        output_model = (Path(test_data_dir) / "../bert_tokenizer.onnx").resolve()
        input_text = ("This is a test sentence",)
        ref_output = [
            np.array([[2, 236, 118, 16, 1566, 875, 643, 3]]),
            np.array([[0, 0, 0, 0, 0, 0, 0, 0]]),
            np.array([[1, 1, 1, 1, 1, 1, 1, 1]]),
        ]
        # tokenizer = transformers.AutoTokenizer.from_pretrained("lordtt13/emo-mobilebert")
        tokenizer_parameters = TokenizerParam(vocab_or_file=os.path.join(test_data_dir, "../bert.vocab"),
                                              do_lower_case=True)
        tokenizer_impl = BertTokenizer(tokenizer_parameters)
        self.create_pipeline_and_run_for_tokenizer(
            tokenizer_impl, "BertTokenizer", tokenizer_parameters, output_model)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        s = ort.InferenceSession(str(output_model), so, providers=["CPUExecutionProvider"])

        result = s.run(None, {s.get_inputs()[0].name: np.array([[*input_text]])})

        self.assertEqual(np.allclose(result[0], ref_output[0]), True)
        self.assertEqual(np.allclose(result[1], ref_output[2]), True)
        self.assertEqual(np.allclose(result[2], ref_output[1]), True)

    def test_hfbert_tokenizer(self):
        output_model = (Path(test_data_dir) / "../hfbert_tokenizer.onnx").resolve()
        ref_output = ([
            np.array([[2, 236, 118, 16, 1566, 875, 643, 3, 236, 118, 978, 1566, 875, 643, 3]]),
            np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]),
            np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
        ]
        )
        # tokenizer = transformers.AutoTokenizer.from_pretrained("lordtt13/emo-mobilebert")
        tokenizer_parameters = TokenizerParam(vocab_or_file=os.path.join(test_data_dir, "../hfbert.vocab"),
                                              do_lower_case=True, is_sentence_pair=True)
        input_text = ("This is a test sentence", "This is another test sentence")
        tokenizer_impl = BertTokenizer(tokenizer_parameters)
        self.create_pipeline_and_run_for_tokenizer(
            tokenizer_impl, "HfBertTokenizer", tokenizer_parameters, output_model)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        s = ort.InferenceSession(str(output_model), so, providers=["CPUExecutionProvider"])

        result = s.run(None, {s.get_inputs()[0].name: np.array([[input_text[0], input_text[1]]])})

        self.assertEqual(np.allclose(result[0], ref_output[0]), True)
        self.assertEqual(np.allclose(result[1], ref_output[2]), True)
        self.assertEqual(np.allclose(result[2], ref_output[1]), True)

    def test_qatask_with_tokenizer(self):
        output_model = (Path(test_data_dir) / "../hfbert_tokenizer.onnx").resolve()
        ref_output = [np.array(["[CLS]"])]
        # tokenizer = transformers.AutoTokenizer.from_pretrained("lordtt13/emo-mobilebert")
        tokenizer_parameters = TokenizerParam(vocab_or_file=os.path.join(test_data_dir, "../hfbert.vocab"),
                                              do_lower_case=True, is_sentence_pair=True)
        input_text = ("This is a test sentence", "This is another test sentence")
        tokenizer_impl = BertTokenizer(tokenizer_parameters)
        self.create_pipeline_and_run_for_tokenizer(
            tokenizer_impl, "HfBertTokenizer_with_decoder", tokenizer_parameters, output_model)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        s = ort.InferenceSession(str(output_model), so, providers=["CPUExecutionProvider"])

        result = s.run(None, {s.get_inputs()[0].name: np.array([[input_text[0], input_text[1]]])})

        self.assertEqual(result[0][0], ref_output[0][0])

    # Corner Case
    def test_debug_step(self):
        import onnx

        create_named_value = pre_post_processing.utils.create_named_value

        # multiple DebugSteps are stringed together
        input_model_path = os.path.join(test_data_dir, "pytorch_super_resolution.onnx")
        inputs = [create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]
        pipeline = pre_post_processing.PrePostProcessor(inputs)
        # each Debug step adds a new model output
        post_processing = [pre_post_processing.Debug(1), pre_post_processing.Debug(1), pre_post_processing.Debug(1)]

        pipeline.add_post_processing(post_processing)
        input_model = onnx.load(input_model_path)
        new_model = pipeline.run(input_model)

        self.assertEqual(len(new_model.graph.output), len(input_model.graph.output) + len(post_processing))

    def draw_boxes_on_image(self, output_model, test_boxes):
        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        ort_sess = ort.InferenceSession(str(output_model), providers=['CPUExecutionProvider'], sess_options=so)
        image = np.frombuffer(open(Path(test_data_dir)/'wolves.jpg', 'rb').read(), dtype=np.uint8)

        return ort_sess.run(None, {'image': image, "boxes_in": test_boxes})[0]

    def test_draw_box_crop_pad(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (Path(test_data_dir) / "../draw_bounding_box.onnx").resolve()
        test_boxes = [np.array([[220.0, 180.0, 450.0, 380.0, 0.5, 0.0]], dtype=np.float32),
                      np.array([[35.0, 200.0, 220.0, 340.0, 0.5, 0.0]], dtype=np.float32)]
        ref_img = ['wolves_with_box_crop.jpg', 'wolves_with_box_pad.jpg']
        for idx, is_crop in enumerate([True, False]):
            output_img = (Path(test_data_dir) / f"../{ref_img[idx]}").resolve()
            create_boxdrawing_model.create_model(output_model, is_crop=is_crop)
            image_ref = np.frombuffer(open(output_img, 'rb').read(), dtype=np.uint8)
            output = self.draw_boxes_on_image(output_model, test_boxes[idx])
            self.assertEqual((image_ref == output).all(), True)

    def test_draw_box_share_border(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (Path(test_data_dir) / "../draw_bounding_box.onnx").resolve()
        test_boxes = np.array([
            [0, 0, 180.0, 150.0, 0.5, 0.0],
            [240, 0, 240.0, 150.0, 0.5, 0.0],
            [0, 240, 140.0, 240.0, 0.5, 0.0],
            [240, 240, 240.0, 240.0, 0.5, 0.0],
        ], dtype=np.float32)

        create_boxdrawing_model.create_model(output_model, mode="XYWH")
        output = self.draw_boxes_on_image(output_model, test_boxes)

        output_img = (Path(test_data_dir) / f"../wolves_with_box_share_borders.jpg").resolve()
        image_ref = np.frombuffer(open(output_img, 'rb').read(), dtype=np.uint8)
        self.assertEqual((image_ref == output).all(), True)

    def test_draw_box_off_boundary_box(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (Path(test_data_dir) / "../draw_bounding_box.onnx").resolve()
        test_boxes = np.array([
            [20, 20, 180.0, 150.0, 0.5, 0.0],
            [340, 0, 240.0, 150.0, 0.5, 0.0],
            [0, 440, 140.0, 440.0, 0.5, 0.0],
            [440, 440, 240.0, 440.0, 0.5, 0.0],
        ], dtype=np.float32)

        create_boxdrawing_model.create_model(output_model, mode="CENTER_XYWH")
        output = self.draw_boxes_on_image(output_model, test_boxes)

        output_img = (Path(test_data_dir) / f"../wolves_with_box_off_boundary_box.jpg").resolve()
        image_ref = np.frombuffer(open(output_img, 'rb').read(), dtype=np.uint8)
        self.assertEqual((image_ref == output).all(), True)

    def test_draw_box_more_box_by_class_than_colors(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (Path(test_data_dir) / "../draw_bounding_box.onnx").resolve()
        test_boxes = np.array([
            [0, 0, 180.0, 150.0, 0.15, 10.0],
            [240, 0, 140.0, 150.0, 0.25, 1.0],
            [0, 240, 140.0, 240.0, 0.35, 2.0],
            [12, 41, 140.0, 140.0, 0.45, 3.0],
            [234, 23, 140.0, 140.0, 0.55, 4.0],
            [64, 355, 140.0, 140.0, 0.65, 5.0],
            [412, 140, 140.0, 140.0, 0.75, 6.0],
            [99, 300, 140.0, 140.0, 0.85, 7.0],
            [199, 200, 140.0, 40.0, 0.95, 8.0],
            [319, 90, 140.0, 40.0, 0.5, 9.0],
            [129, 130, 140.0, 40.0, 0.5, 10.0],
            [239, 190, 140.0, 40.0, 0.5, 11.0],
            [49, 240, 140.0, 40.0, 0.5, 12.0],
            [259, 290, 140.0, 40.0, 0.99, 10.0],
        ], dtype=np.float32)

        create_boxdrawing_model.create_model(output_model, mode="XYWH", colour_by_classes=True)
        output = self.draw_boxes_on_image(output_model, test_boxes)

        output_img = (Path(test_data_dir) / f"../wolves_with_box_more_box_than_colors.jpg").resolve()
        image_ref = np.frombuffer(open(output_img, 'rb').read(), dtype=np.uint8)
        self.assertEqual((image_ref == output).all(), True)

    def test_draw_box_more_box_by_score_than_colors(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (Path(test_data_dir) / "../draw_bounding_box.onnx").resolve()
        test_boxes = np.array([
            [0, 0, 180.0, 150.0, 0.15, 0.0],
            [240, 0, 140.0, 150.0, 0.25, 0.0],
            [0, 240, 140.0, 240.0, 0.35, 0.0],
            [12, 41, 140.0, 140.0, 0.45, 0.0],
            [234, 23, 140.0, 140.0, 0.55, 0.0],
            [64, 355, 140.0, 140.0, 0.65, 0.0],
            [412, 140, 140.0, 140.0, 0.75, 0.0],
            [99, 300, 140.0, 140.0, 0.85, 0.0],
            [199, 200, 140.0, 140.0, 0.95, 0.0],
            [319, 90, 140.0, 140.0, 0.5, 0.0],
            [129, 130, 140.0, 140.0, 0.5, 0.0],
            [239, 190, 140.0, 140.0, 0.5, 0.0],
            [49, 240, 140.0, 140.0, 0.5, 0.0],
            [259, 290, 140.0, 140.0, 0.98, 0.0],
            [10, 10, 340.0, 340.0, 0.99, 0.0],
        ], dtype=np.float32)

        create_boxdrawing_model.create_model(output_model, mode="XYWH", colour_by_classes=False)
        output = self.draw_boxes_on_image(output_model, test_boxes)

        output_img = (Path(test_data_dir) / f"../wolves_with_box_more_box_than_colors_score.jpg").resolve()
        image_ref = np.frombuffer(open(output_img, 'rb').read(), dtype=np.uint8)
        self.assertEqual((image_ref == output).all(), True)

    # a box with higher score should be drawn over a box with lower score

    def test_draw_box_overlapping_with_priority(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (Path(test_data_dir) / "../draw_bounding_box.onnx").resolve()
        test_boxes = np.array([
            [0, 0, 240, 240, 0.5, 0.0],
            [40, 40, 240, 240, 0.5, 0.0],
            [100, 100, 240.0, 240.0, 0.5, 0.0],
            [140, 140, 240.0, 240.0, 0.5, 0.0],
        ], dtype=np.float32)

        create_boxdrawing_model.create_model(output_model, mode="XYWH")
        output = self.draw_boxes_on_image(output_model, test_boxes)

        output_img = (Path(test_data_dir) / f"../wolves_with_box_overlapping.jpg").resolve()
        image_ref = np.frombuffer(open(output_img, 'rb').read(), dtype=np.uint8)
        self.assertEqual((image_ref == output).all(), True)

    def test_draw_box_with_large_thickness(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (Path(test_data_dir) / "../draw_bounding_box.onnx").resolve()
        test_boxes = np.array([
            [0, 0, 40, 40, 0.5, 0.0],
            [40, 40, 40, 40, 0.5, 0.0],
            [478, 478, 480.0, 480.0, 0.5, 0.0],
            [140, 140, 40.0, 40.0, 0.5, 0.0],
        ], dtype=np.float32)

        create_boxdrawing_model.create_model(output_model, mode="XYXY", thickness=1000)
        output = self.draw_boxes_on_image(output_model, test_boxes)

        output_img = (Path(test_data_dir) / f"../wolves_with_solid_box.jpg").resolve()
        image_ref = np.frombuffer(open(output_img, 'rb').read(), dtype=np.uint8)
        self.assertEqual((image_ref == output).all(), True)

    def create_pipeline_and_run_for_nms(self, output_model: Path, length: int,
                                        iou_threshold: float = 0.5,
                                        score_threshold: float = 0.7,
                                        max_detections: int = 10):
        import onnx
        create_named_value = pre_post_processing.utils.create_named_value

        inputs = [create_named_value("box_and_score", onnx.TensorProto.FLOAT, ["num_boxes", length])]

        onnx_opset = 16
        pipeline = pre_post_processing.PrePostProcessor(inputs, onnx_opset)

        pipeline.add_post_processing([
            SplitOutBoxAndScore(num_classes=1),
            SelectBestBoundingBoxesByNMS(iou_threshold=iou_threshold, score_threshold=score_threshold,
                                         max_detections=max_detections),
        ])

        graph_def = onnx.parser.parse_graph(
            f"""\
            identity (float[num_boxes,{length}] _input)
                => (float[num_boxes,{length}] _output)  
            {{
                _output = Identity(_input)
            }}
        """)

        onnx_import = onnx.helper.make_operatorsetid('', onnx_opset)
        ir_version = onnx.helper.find_min_ir_version_for([onnx_import])
        input_model = onnx.helper.make_model_gen_version(graph_def, opset_imports=[onnx_import], ir_version=ir_version)

        new_model = pipeline.run(input_model)
        onnx.save_model(new_model, output_model)

    def test_NMS_and_drawing_box_without_confOfObj(self):
        output_model = (Path(test_data_dir) / "../nms.onnx").resolve()
        self.create_pipeline_and_run_for_nms(output_model, iou_threshold=0.9, length=5)
        input_data = [
            [0, 0, 240, 240, 0.75],
            [10, 10, 240, 240, 0.75],
            [50, 50, 240, 240, 0.75],
            [150, 150, 240, 240, 0.75],
        ]
        input_data = np.array(input_data, dtype=np.float32)
        output_data_ref = np.concatenate([input_data, np.zeros((4, 1))], axis=-1)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        ort_sess = ort.InferenceSession(str(output_model), providers=['CPUExecutionProvider'], sess_options=so)

        out = ort_sess.run(None, {'_input': input_data})[0]

        self.assertEqual((out == output_data_ref).all(), True)

    def test_NMS_and_drawing_box_with_confOfObj(self):
        output_model = (Path(test_data_dir) / "../nms.onnx").resolve()
        self.create_pipeline_and_run_for_nms(output_model, iou_threshold=0.9, score_threshold=0.5, length=6)
        input_data = [
            [0, 0, 240, 240, 0.75, 0.9],
            [10, 10, 240, 240, 0.75, 0.9],
            [50, 50, 240, 240, 0.75, 0.9],
            [150, 150, 240, 240, 0.75, 0.9],
        ]
        input_data = np.array(input_data, dtype=np.float32)
        output_data_ref = np.concatenate([input_data[:,0:-1], np.zeros((4, 1))], axis=-1)
        output_data_ref[:, -2] = 0.67499

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        ort_sess = ort.InferenceSession(str(output_model), providers=['CPUExecutionProvider'], sess_options=so)

        out = ort_sess.run(None, {'_input': input_data})[0]
        self.assertEqual(np.abs(out-output_data_ref).max() < 10e-6, True)

    def test_NMS_and_drawing_box_iou_and_score_threshold(self):
        output_model = (Path(test_data_dir) / "../nms.onnx").resolve()
        
        def get_model_output():
            input_data = [
                [0, 0, 240, 240, 0.75, 0.9],
                [10, 10, 240, 240, 0.85, 0.9],
                [50, 50, 240, 240, 0.95, 0.9],
                [150, 150, 240, 240, 0.99, 0.99],
            ]
            input_data = np.array(input_data, dtype=np.float32)

            so = ort.SessionOptions()
            so.register_custom_ops_library(get_library_path())
            ort_sess = ort.InferenceSession(str(output_model), providers=['CPUExecutionProvider'], sess_options=so)

            out = ort_sess.run(None, {'_input': input_data})[0]
            return out
            
        expected_size = [24,12,6,18,12,6,18,12,6,]
        idx = 0
        for iou_threshold in [0.9, 0.75, 0.5]:
            for score_threshold in [0.5, 0.8, 0.9]:
                self.create_pipeline_and_run_for_nms(
                    output_model, iou_threshold=iou_threshold, score_threshold=score_threshold, length=6)
                out = get_model_output()
                self.assertEqual(out.size, expected_size[idx])
                idx += 1
        
    def test_FastestDet(self):
        # https://github.com/dog-qiuqiu/FastestDet
        # a minor fix is to accommodate output with yolo output format, including bounding box regression inside.
        input_model = os.path.join(test_data_dir, "FastestDet.onnx")
        output_model = os.path.join(test_data_dir, "FastestDet.updated.onnx")
        input_image_path = os.path.join(test_data_dir, "wolves.jpg")

        add_ppp.yolo_detection(Path(input_model), Path(output_model),input_shape=(352,352))

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        ort_sess = ort.InferenceSession(str(output_model), providers=['CPUExecutionProvider'], sess_options=so)
        image = np.frombuffer(open(Path(test_data_dir)/'wolves.jpg', 'rb').read(), dtype=np.uint8)

        output = ort_sess.run(None, {'image': image})[0]
        output_img = (Path(test_data_dir) / f"../wolves_with_fastestDet.jpg").resolve()
        image_ref = np.frombuffer(open(output_img, 'rb').read(), dtype=np.uint8)
        self.assertEqual((image_ref == output).all(), True)

if __name__ == "__main__":
    unittest.main()
