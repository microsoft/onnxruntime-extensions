# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import io
import numpy as np
import onnxruntime as ort
import os

from PIL import Image
from pathlib import Path
from packaging import version
# NOTE: This assumes you have created an editable pip install for onnxruntime_extensions by running
# `pip install -e .` from the repo root.
from onnxruntime_extensions import get_library_path
from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
from onnxruntime_extensions.tools import add_HuggingFace_CLIPImageProcessor_to_model as add_clip_feature
from onnxruntime_extensions.tools import pre_post_processing as pre_post_processing
from onnxruntime_extensions.tools.pre_post_processing import *  # noqa


script_dir = os.path.dirname(os.path.realpath(__file__))
ort_ext_root = os.path.abspath(os.path.join(script_dir, ".."))
test_data_dir = os.path.join(ort_ext_root, "test", "data", "ppp_vision")


def compare_two_images_mse(image1, image2):
    # decoding it firstly to avoid any format issues
    image1 = Image.open(io.BytesIO(image1))
    image2 = Image.open(io.BytesIO(image2))
    if image1.size != image2.size:
        return 10   # arbitrary large value
    # check if the images are similar by MSE
    return np.mean(np.square(np.array(image1) - np.array(image2)))


def load_image_file(file_path):
    with open(file_path, "rb") as f:
        return f.read()


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

@unittest.skipIf(version.parse(ort.__version__) < version.parse("1.13"), "only supported in ort 1.13 and above")
class TestToolsAddPrePostProcessingToModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp4onnx = Path(test_data_dir).parent / "temp_4onnx"
        temp4onnx.mkdir(parents=True, exist_ok=True)
        cls.temp4onnx = temp4onnx

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

    def test_pytorch_mobilenet_using_clip_feature(self):
        input_model = os.path.join(test_data_dir, "pytorch_mobilenet_v2.onnx")
        output_model = os.path.join(test_data_dir, "pytorch_mobilenet_v2.updated.onnx")
        input_image_path = os.path.join(test_data_dir, "wolves.jpg")

        add_clip_feature.clip_image_processor(Path(input_model), Path(output_model), opset=16, do_resize=True, 
                                              do_center_crop=True, do_normalize=True, do_rescale=True,
                                              do_convert_rgb=True, size=256, crop_size=224, 
                                              rescale_factor=1/255, image_mean=[0.485, 0.456, 0.406],
                                              image_std=[0.229, 0.224, 0.225])

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
            scores = np.squeeze(probabilities)  # remove batch dim

            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()

            probabilities = softmax(scores)
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
        output_model = (self.temp4onnx / "sentencePiece.onnx").resolve()

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
        output_model = (self.temp4onnx / "bert_tokenizer.onnx").resolve()
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
        output_model = (self.temp4onnx / "hfbert_tokenizer.onnx").resolve()
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

    def test_hfbert_tokenizer_optional_output(self):
        output_model = (self.temp4onnx / "hfbert_tokenizer_optional_output.onnx").resolve()

        ref_output = ([
            np.array([[2, 236, 118, 16, 1566, 875, 643, 3, 236, 118, 978, 1566, 875, 643, 3]]),
            np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
        ]
        )
        # tokenizer = transformers.AutoTokenizer.from_pretrained("lordtt13/emo-mobilebert")
        tokenizer_parameters = TokenizerParam(vocab_or_file=os.path.join(test_data_dir, "../hfbert.vocab"),
                                              do_lower_case=True, is_sentence_pair=True)
        input_text = ("This is a test sentence", "This is another test sentence")
        tokenizer_impl = BertTokenizer(tokenizer_parameters, True)
        self.create_pipeline_and_run_for_tokenizer(
            tokenizer_impl, "HfBertTokenizer", tokenizer_parameters, output_model)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        s = ort.InferenceSession(str(output_model), so, providers=["CPUExecutionProvider"])

        result = s.run(None, {s.get_inputs()[0].name: np.array([[input_text[0], input_text[1]]])})

        self.assertEqual(len(result), 2)

        self.assertEqual(np.allclose(result[0], ref_output[0]), True)
        self.assertEqual(np.allclose(result[1], ref_output[1]), True)

    def test_qatask_with_tokenizer(self):
        output_model = (self.temp4onnx / "hfbert_tokenizer.onnx").resolve()
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
        image = np.frombuffer(load_image_file(Path(test_data_dir)/'wolves.jpg'), dtype=np.uint8)

        return ort_sess.run(None, {'image': image, "boxes_in": test_boxes})[0]

    def test_draw_box_crop_pad(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (self.temp4onnx / "draw_bounding_box.onnx").resolve()
        test_boxes = [np.array([[220.0, 180.0, 450.0, 380.0, 0.5, 0.0]], dtype=np.float32),
                      np.array([[35.0, 200.0, 220.0, 340.0, 0.5, 0.0]], dtype=np.float32)]
        ref_img = ['wolves_with_box_crop.jpg', 'wolves_with_box_pad.jpg']
        for idx, is_crop in enumerate([True, False]):
            output_img = (Path(test_data_dir) / f"../{ref_img[idx]}").resolve()
            create_boxdrawing_model.create_model(output_model, is_crop=is_crop)
            image_ref = np.frombuffer(load_image_file(output_img), dtype=np.uint8)
            output = self.draw_boxes_on_image(output_model, test_boxes[idx])
            self.assertLess(compare_two_images_mse(image_ref, output), 0.13)

    def test_draw_box_share_border(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (self.temp4onnx / "draw_bounding_box.onnx").resolve()
        test_boxes = np.array([
            [0, 0, 180.0, 150.0, 0.5, 0.0],
            [240, 0, 240.0, 150.0, 0.5, 0.0],
            [0, 240, 140.0, 240.0, 0.5, 0.0],
            [240, 240, 240.0, 240.0, 0.5, 0.0],
        ], dtype=np.float32)

        create_boxdrawing_model.create_model(output_model, mode="XYWH")
        output = self.draw_boxes_on_image(output_model, test_boxes)

        output_img = (Path(test_data_dir) / f"../wolves_with_box_share_borders.jpg").resolve()
        image_ref = np.frombuffer(load_image_file(output_img), dtype=np.uint8)
        self.assertLess(compare_two_images_mse(image_ref, output), 0.1)

    def test_draw_box_off_boundary_box(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (self.temp4onnx / "draw_bounding_box.onnx").resolve()
        test_boxes = np.array([
            [20, 20, 180.0, 150.0, 0.5, 0.0],
            [340, 0, 240.0, 150.0, 0.5, 0.0],
            [0, 440, 140.0, 440.0, 0.5, 0.0],
            [440, 440, 240.0, 440.0, 0.5, 0.0],
        ], dtype=np.float32)

        create_boxdrawing_model.create_model(output_model, mode="CENTER_XYWH")
        output = self.draw_boxes_on_image(output_model, test_boxes)

        output_img = (Path(test_data_dir) / f"../wolves_with_box_off_boundary_box.jpg").resolve()
        image_ref = np.frombuffer(load_image_file(output_img), dtype=np.uint8)
        self.assertLess(compare_two_images_mse(image_ref, output), 0.1)

    def test_draw_box_more_box_by_class_than_colors(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (self.temp4onnx / "draw_bounding_box.onnx").resolve()
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
        image_ref = np.frombuffer(load_image_file(output_img), dtype=np.uint8)
        self.assertLess(compare_two_images_mse(image_ref, output), 0.1)

    def test_draw_box_more_box_by_score_than_colors(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (self.temp4onnx / "draw_bounding_box.onnx").resolve()
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
        image_ref = np.frombuffer(load_image_file(output_img), dtype=np.uint8)
        self.assertLess(compare_two_images_mse(image_ref, output), 0.1)

    # a box with higher score should be drawn over a box with lower score

    def test_draw_box_overlapping_with_priority(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (self.temp4onnx / "draw_bounding_box.onnx").resolve()
        test_boxes = np.array([
            [0, 0, 240, 240, 0.5, 0.0],
            [40, 40, 240, 240, 0.5, 0.0],
            [100, 100, 240.0, 240.0, 0.5, 0.0],
            [140, 140, 240.0, 240.0, 0.5, 0.0],
        ], dtype=np.float32)

        create_boxdrawing_model.create_model(output_model, mode="XYWH")
        output = self.draw_boxes_on_image(output_model, test_boxes)

        output_img = (Path(test_data_dir) / f"../wolves_with_box_overlapping.jpg").resolve()
        image_ref = np.frombuffer(load_image_file(output_img), dtype=np.uint8)
        self.assertLess(compare_two_images_mse(image_ref, output), 0.1)

    def test_draw_box_with_large_thickness(self):
        import sys
        sys.path.append(test_data_dir)
        import create_boxdrawing_model

        output_model = (self.temp4onnx / "draw_bounding_box.onnx").resolve()
        test_boxes = np.array([
            [0, 0, 40, 40, 0.5, 0.0],
            [40, 40, 40, 40, 0.5, 0.0],
            [478, 478, 480.0, 480.0, 0.5, 0.0],
            [140, 140, 40.0, 40.0, 0.5, 0.0],
        ], dtype=np.float32)

        create_boxdrawing_model.create_model(output_model, mode="XYXY", thickness=1000)
        output = self.draw_boxes_on_image(output_model, test_boxes)

        output_img = (Path(test_data_dir) / f"../wolves_with_solid_box.jpg").resolve()
        image_ref = np.frombuffer(load_image_file(output_img), dtype=np.uint8)
        self.assertLess(compare_two_images_mse(image_ref, output), 0.1)

    def _create_pipeline_and_run_for_nms(self, output_model: Path,
                                         has_conf_value: bool,
                                         iou_threshold: float = 0.5,
                                         score_threshold: float = 0.7,
                                         max_detections: int = 100,
                                         max_boxes_per_class: int = 100,
                                         num_classes: int = 1):
        import onnx
        create_named_value = pre_post_processing.utils.create_named_value
        length = (5 if has_conf_value else 4) + num_classes
        # [ num_boxes, <4 points for box, optional conf, one score per class> ]
        inputs = [create_named_value("_input", onnx.TensorProto.FLOAT, ["num_boxes", length])]

        onnx_opset = 16
        pipeline = pre_post_processing.PrePostProcessor(inputs, onnx_opset)

        if has_conf_value:
            pipeline.add_post_processing([
                SplitOutBoxAndScoreWithConf(num_classes=num_classes),
                SelectBestBoundingBoxesByNMS(iou_threshold=iou_threshold, score_threshold=score_threshold,
                                             max_boxes_per_class=max_boxes_per_class, max_detections=max_detections),
            ])
        else:
            pipeline.add_post_processing([
                # split the 4 bounding box co-ords from the class scores
                Split(num_outputs=2, axis=-1, splits=[4, num_classes]),
                SelectBestBoundingBoxesByNMS(iou_threshold=iou_threshold, score_threshold=score_threshold,
                                             max_boxes_per_class=max_boxes_per_class, max_detections=max_detections),
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
        output_model = (self.temp4onnx / "nms.onnx").resolve()
        self._create_pipeline_and_run_for_nms(output_model, iou_threshold=0.9, has_conf_value=False)
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
        output_model = (self.temp4onnx / "nms.onnx").resolve()
        self._create_pipeline_and_run_for_nms(output_model, iou_threshold=0.9, score_threshold=0.5, has_conf_value=True)
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
        output_model = (self.temp4onnx / "nms.onnx").resolve()

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
                self._create_pipeline_and_run_for_nms(
                    output_model, iou_threshold=iou_threshold, score_threshold=score_threshold, has_conf_value=True)
                out = get_model_output()
                self.assertEqual(out.size, expected_size[idx])
                idx += 1

    def test_NMS_max_detections(self):
        def run_test(max_per_class, max_overall):

            output_model = (self.temp4onnx / "nms_max_det.onnx").resolve()
            self._create_pipeline_and_run_for_nms(output_model, has_conf_value=False, iou_threshold=0.95, num_classes=2,
                                                  max_boxes_per_class=max_per_class, max_detections=max_overall)
            input_data = [
                [25, 25, 10, 10, 0.75, 0.85],
                [100, 100, 10, 10, 0.91, 0.72],
                [25, 150, 10, 10, 0.83, 0.93],
                [150, 150, 10, 10, 0.87, 0.77],
            ]

            input_data = np.array(input_data, dtype=np.float32)

            num_classes = 2
            # max results is returning both classes for every bounding box
            num_to_select = min(max_overall, num_classes * len(input_data))
            num_selected = 0
            num_selected_per_class = [0 for i in range(0, num_classes)]

            results_expected = [[] for i in range(0, num_classes)]
            scores = input_data[:, -2:].copy()  # copy as we set values to 0 as we go along
            # pick the initial set of results based on score
            cur_result = 0
            while num_selected < num_to_select and cur_result < scores.size:
                cur_result += 1  # we may run out of results before we select enough
                expected = []

                best_score = scores.max()
                idx = int(scores.argmax() / num_classes)  # find row best score came from. num_classes entries per row.
                selected_class = np.where(scores[idx] == best_score)[0][0]  # find index of best score
                scores[idx][selected_class] = 0.  # set the score to 0 so it doesn't get selected again

                if num_selected_per_class[selected_class] == max_per_class:
                    continue

                box = np.array(input_data[idx][:4])
                expected += box.tolist()
                expected.append(best_score)
                expected.append(selected_class)

                results_expected[selected_class].append(expected)
                num_selected_per_class[selected_class] += 1
                num_selected += 1

            so = ort.SessionOptions()
            so.register_custom_ops_library(get_library_path())
            ort_sess = ort.InferenceSession(str(output_model), providers=['CPUExecutionProvider'], sess_options=so)

            # flatten the per-class entries from
            #   {num_classes, num selected results, result size} to {num_classes * num_results, result size}
            results_expected = [np.asarray(entry) for entry in results_expected if len(entry) > 0]
            results_expected = np.concatenate(results_expected).reshape((-1, 6))

            outputs = ort_sess.run(None, {'_input': input_data})
            results_actual = outputs[0]

            self.assertEqual(results_expected.shape, results_actual.shape)
            compared = np.isclose(results_expected, results_actual)
            self.assertTrue(compared.all(),
                            msg=f"\nExpected={results_expected}\nActual={results_actual}\nCompared={compared}")

        run_test(100, 3)  # max overall trims
        run_test(1, 100)  # max per class trims
        run_test(1, 1)  # max per class and max overall trim

    # Create pipeline to run NMS and scaling.
    # Scaling should handle converting back to co-ordinates in the original image that was resized and letterboxed
    def _create_pipeline_and_run_for_nms_and_scaling(self, output_model: Path,
                                                     orig_image_shape: List[int],  # 3 dims, HWC or CHW
                                                     resized_image_shape: List[int],
                                                     letterboxed_image_shape: List[int],
                                                     num_classes: int = 1,
                                                     has_key_points: bool = False,
                                                     key_points_have_conf: bool = False,
                                                     ):

        # channels are 3 so infer layout from shape
        layout = "HWC" if orig_image_shape[-1] == 3 else "CHW"

        mask_data_size = 0
        if has_key_points:
            # 3 results of x and y. optional conf in each result
            mask_data_size = 3 * (3 if key_points_have_conf else 2)

        result_data_size = 4 + num_classes + mask_data_size

        # create graph to provide outputs for post-processing
        inputs = [utils.create_named_value("results", onnx.TensorProto.FLOAT, ["num_boxes", result_data_size]),
                  utils.create_named_value("orig_img", onnx.TensorProto.UINT8, orig_image_shape),
                  utils.create_named_value("resized_img", onnx.TensorProto.UINT8, resized_image_shape),
                  utils.create_named_value("letterboxed_img", onnx.TensorProto.UINT8, letterboxed_image_shape),
                  ]

        graph_input_strings = [f"float[num_boxes, {result_data_size}] results",
                               f"uint8[{','.join([str(i) for i in orig_image_shape])}] orig_img",
                               f"uint8[{','.join([str(i) for i in resized_image_shape])}] resized_img",
                               f"uint8[{','.join([str(i) for i in letterboxed_image_shape])}] letterboxed_img",
                               ]

        graph_output_strings = [s + "_out" for s in graph_input_strings]
        graph_nodes = "\n".join([f"{input.name}_out = Identity({input.name})" for input in inputs])

        onnx_opset = 16

        graph_text = \
            f"""pass_through ({', '.join(graph_input_strings)}) => ({', '.join(graph_output_strings)})             
            {{
                {graph_nodes}
            }}"""

        graph_def = onnx.parser.parse_graph(graph_text)

        onnx_import = onnx.helper.make_operatorsetid('', onnx_opset)
        ir_version = onnx.helper.find_min_ir_version_for([onnx_import])
        input_model = onnx.helper.make_model_gen_version(graph_def, opset_imports=[onnx_import], ir_version=ir_version)

        # if there is mask data containing keypoints we need to split that out
        splits = [4, num_classes]
        if has_key_points:
            splits.append(mask_data_size)

        pipeline = pre_post_processing.PrePostProcessor(inputs, onnx_opset)

        post_processing = [
            # pass through model inputs via a Step so the original, resized and letterboxed shapes are available
            # to use in the IoMapEntry for scaling
            Identity(num_inputs=4, name="InputsPassThrough"),
            Split(num_outputs=len(splits), axis=1, splits=splits),
            SelectBestBoundingBoxesByNMS(iou_threshold=0.7, score_threshold=0.25, has_mask_data=has_key_points),
            # Scale boxes and key point coords back to original image. Mask data has 3 key points per box.
            (ScaleNMSBoundingBoxesAndKeyPoints(num_key_points=3, layout=layout),
             [
                 # A default connection from SelectBestBoundingBoxesByNMS for input 0
                 # A connection from original image
                 # A connection from the resized image
                 # A connection from the LetterBoxed image
                 # We use the images to calculate the scale factor and offset.
                 # With scale and offset, we can scale the bounding box and key points back to the original image.
                 utils.IoMapEntry("InputsPassThrough", producer_idx=1, consumer_idx=1),
                 utils.IoMapEntry("InputsPassThrough", producer_idx=2, consumer_idx=2),
                 utils.IoMapEntry("InputsPassThrough", producer_idx=3, consumer_idx=3),
             ]),
        ]

        pipeline.add_post_processing(post_processing)

        new_model = pipeline.run(input_model)
        onnx.save_model(new_model, str(output_model))

    def _run_nms_scaling_test(self, channels_last: bool = True, num_classes: int = 1,
                              has_key_points: bool = False, key_points_have_conf: bool = False):
        model_name = (f"nms_{'HWC' if channels_last else 'CHW'}_c{num_classes}_"
                      f"kp{has_key_points}_kpc{key_points_have_conf}")
        output_model = (self.temp4onnx / f"{model_name}.onnx").resolve()

        if channels_last:
            h_dim, w_dim = 0, 1
            orig_image_shape = [400, 500, 3]  # HWC
            resized_image_shape = [320, 400, 3]  # Resize to not_smaller 400 x 400
            letterboxed_image_shape = [400, 400, 3]  # letterbox to 400 x 400
        else:
            h_dim, w_dim = 1, 2
            orig_image_shape = [3, 400, 500]
            resized_image_shape = [3, 320, 400]
            letterboxed_image_shape = [3, 400, 400]

        scale_ratio = 500 / 400  # we kept the aspect ratio
        # width and height padding to apply to first 2 points of box as format is XYWH
        # / 2 as we
        half_pad_h = (letterboxed_image_shape[h_dim] - resized_image_shape[h_dim]) / 2
        half_pad_w = (letterboxed_image_shape[w_dim] - resized_image_shape[w_dim]) / 2
        letterbox_padding = np.array([half_pad_w, half_pad_h, 0, 0], dtype=np.float32)

        # default score threshold is 0.25 so this will ensure no results are thrown away due to the score
        np.random.seed(123)
        # scores0 = np.random.uniform(low=0.5, high=1.0, size=num_classes)
        # scores1 = scores0 - 0.1  # first result should win if picking a single result and be first in NMS output
        # scores = [scores0, scores1]
        scores = np.random.uniform(low=0.5, high=1.0, size=(2, num_classes))

        if has_key_points:
            if key_points_have_conf:
                keypoints = [[5., 5., .8, 10., 10., .8, 60., 60., .9],
                             [60., 60., .9, 80., 80., .6, 150., 120., .5]]
            else:
                keypoints = [[5., 5., 10., 10., 60., 60.],
                             [60., 60., 80., 80., 150., 120.]]
        else:
            keypoints = [[], []]

        # 4 for box, num_classes scores, key point data
        input_data = [
            [50., 50., 100., 100., *scores[0], *keypoints[0]],
            [80., 80., 100., 100., *scores[1], *keypoints[1]],
        ]
        input_data = np.array(input_data, dtype=np.float32)

        model_inputs = {
            "results": input_data,
            "orig_img": np.ones(orig_image_shape, dtype=np.uint8),
            "resized_img": np.ones(resized_image_shape, dtype=np.uint8),
            "letterboxed_img": np.ones(letterboxed_image_shape, dtype=np.uint8),
        }

        # for each result, manually scale box and keypoints to validate. check for correct class and score info.
        # we aren't limiting results based on max classes per box or max overall matches so we expect all classes
        # to be returned as results for both bounding boxes.
        # the NMS output is sorted by class first and score second, so we assemble the results on a per-class basis
        # and flatten to compare with the actual results
        results_expected = [[] for i in range(0, num_classes)]
        num_selected = 0
        while num_selected < num_classes * len(input_data):
            expected = []

            best_score = scores.max()
            idx = int(scores.argmax() / num_classes)  # find row best score came from. num_classes entry per row.

            box = np.array(input_data[idx][:4])
            box -= letterbox_padding
            box *= scale_ratio
            expected += box.tolist()

            selected_class = np.where(scores[idx] == best_score)[0][0]  # find index of best score
            expected.append(best_score)
            expected.append(selected_class)

            # set the score to 0 so it doesn't get selected again
            scores[idx][selected_class] = 0.

            # keypoints
            values_per_entry = 3 if key_points_have_conf else 2
            for kp_idx, kp in enumerate(input_data[idx][4 + num_classes:]):

                if kp_idx % values_per_entry == 0:
                    # x coord
                    expected.append((kp - letterbox_padding[0]) * scale_ratio)
                elif kp_idx % values_per_entry == 1:
                    # y coord
                    expected.append((kp - letterbox_padding[1]) * scale_ratio)
                else:
                    assert key_points_have_conf
                    # confidence score should match input
                    expected.append(keypoints[idx][kp_idx])

            results_expected[selected_class].append(expected)
            num_selected += 1

        self._create_pipeline_and_run_for_nms_and_scaling(
            output_model, orig_image_shape, resized_image_shape, letterboxed_image_shape,
            num_classes, has_key_points, key_points_have_conf)

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        ort_sess = ort.InferenceSession(str(output_model), providers=['CPUExecutionProvider'], sess_options=so)

        outputs = ort_sess.run(None, model_inputs)
        results_actual = outputs[0]
        # flatten the per-class entries. we are returning results for all classes of both bounding boxes so should be
        # equal to scores.size
        #   {num_classes, num results, result size} to {num_classes * num_results, result size}
        results_expected = np.asarray(results_expected).reshape((scores.size, -1))
        self.assertEqual(results_expected.shape, results_actual.shape)

        compared = np.isclose(results_expected, results_actual)
        self.assertTrue(compared.all(),
                        msg=f"\nExpected={results_expected}\nActual={results_actual}\nCompared={compared}")

    def test_NMS_with_scaling_and_keypoints(self):
        """
        Test selecting bounding boxes with NMS and scaling the results.
        Include testing of when there are key points in mask data in the results (used by pose models)
        """
        for channels_last in [True, False]:
            for num_classes in [1, 4]:
                for has_key_points in [True, False]:
                    # it only makes sense to have keypoints when there's a single class as the keypoints are
                    # per bounding box. e.g. if you have a bounding box and classes of person and dog, each class would
                    # require totally different keypoints
                    if not has_key_points or num_classes == 1:
                        msg = (f"Running test with layout={'HWC' if channels_last else 'CHW'} "
                               f"num_classes={num_classes} has_key_points={has_key_points}")
                        print(msg)
                        self._run_nms_scaling_test(channels_last, num_classes, has_key_points)
                        if has_key_points:
                            key_points_have_conf = True
                            print(msg + " key_points_have_conf=True")
                            self._run_nms_scaling_test(channels_last, num_classes, has_key_points, key_points_have_conf)

    def test_FastestDet(self):
        # https://github.com/dog-qiuqiu/FastestDet
        # a minor fix is to accommodate output with yolo output format, including bounding box regression inside.
        input_model = os.path.join(test_data_dir, "FastestDet.onnx")
        output_model = os.path.join(test_data_dir, "FastestDet.updated.onnx")
        input_image_path = os.path.join(test_data_dir, "wolves.jpg")

        add_ppp.yolo_detection(Path(input_model), Path(output_model), output_format='png', input_shape=(352, 352))

        so = ort.SessionOptions()
        so.register_custom_ops_library(get_library_path())
        ort_sess = ort.InferenceSession(str(output_model), providers=['CPUExecutionProvider'], sess_options=so)
        image = np.frombuffer(load_image_file(input_image_path), dtype=np.uint8)

        output = ort_sess.run(None, {'image': image})[0]
        output_img = (Path(test_data_dir) / "../wolves_with_fastestDet.png").resolve()
        # output.tofile(str(output_img) + "actual.png")

        image_ref = np.frombuffer(load_image_file(output_img), dtype=np.uint8)
        self.assertLess(compare_two_images_mse(image_ref, output), 0.1)


if __name__ == "__main__":
    unittest.main()
