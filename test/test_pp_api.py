import os
import json
import tempfile
import requests
import unittest
import numpy as np

from PIL import Image
from onnxruntime_extensions import util

# uncomment it if there is a protobuf version mismatch error
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
is_pp_api_available = False
hf_token_id = None
try:
    from transformers import AutoImageProcessor, AutoTokenizer
    from onnxruntime_extensions import pp_api

    is_pp_api_available = True
    hf_token_id = os.environ.get("HF_TOKEN", None)
except ImportError:
    pass

openai_image_mean = np.array([0.48145466, 0.4578275, 0.40821073])
openai_image_std = np.array([0.26862954, 0.26130258, 0.27577711])

phi4_image_mean = np.array([0.5, 0.5, 0.5])
phi4_image_std = np.array([0.5, 0.5, 0.5])


def regen_image(arr, mean, std):
    # Reverse normalization
    array = arr * std + mean

    # Clip the values to [0, 1] range
    array = np.clip(array, 0, 1)

    # Convert to [0, 255] range and uint8 type
    array = (array * 255).astype(np.uint8)

    # Convert NumPy array to PIL Image
    image = Image.fromarray(array)
    return image


@unittest.skipIf(not is_pp_api_available, "pp_api is not available")
class TestPPAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if os.path.exists("/temp"):
            cls.temp_dir = "/temp"
        elif os.path.exists("/tmp"):
            cls.temp_dir = "/tmp"
        else:
            cls.temp_dir = tempfile.mkdtemp()
            print(f"Created temp dir: {cls.temp_dir}")

    def test_CLIP_image_processing(self):
        model_id = "openai/clip-vit-large-patch14"
        image_list = [
            "data/processor/australia.jpg",
            "data/processor/passport.png",
            "data/processor/exceltable.png",
        ]
        (image, image2, image3) = [Image.open(util.get_test_data_file(f)) for f in image_list]

        processor = AutoImageProcessor.from_pretrained(model_id)
        inputs = processor.preprocess([image, image2, image3], return_tensors="np")
        print({k: v.shape if k == "pixel_values" else v for k, v in inputs.items()})

        expected_images = inputs["pixel_values"]
        for i in range(len(expected_images)):
            expected = expected_images[i]
            e_image = regen_image(np.transpose(expected, (1, 2, 0)), openai_image_mean, openai_image_std)
            e_image.save(f"{self.temp_dir}/CLIP_e_{i}.png")

        ort_processor = pp_api.ImageProcessor("test/data/processor/clip_image.json")
        inputs = ort_processor.pre_process([util.get_test_data_file(f) for f in image_list])
        print(ort_processor.to_numpy(inputs, 0).shape)
        actual_images = ort_processor.to_numpy(inputs, 0)
        for i in range(len(actual_images)):
            actual = actual_images[i]
            a_image = regen_image(np.transpose(actual, (1, 2, 0)), openai_image_mean, openai_image_std)
            a_image.save(f"{self.temp_dir}/CLIP_a_{i}.png")

    @unittest.skipIf(hf_token_id is None, "HF_TOKEN is not available")
    def test_llama3_2_image_processing(self):
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        url = (
            "https://huggingface.co/datasets/huggingface/"
            "documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
        )
        # save the image to a file in self.temp_dir
        with open(f"{self.temp_dir}/rabbit.jpg", "wb") as f:
            f.write(requests.get(url).content)

        # image = Image.open(requests.get(url, stream=True).raw)
        image_list = [
            f"{self.temp_dir}/rabbit.jpg",
            "test/data/processor/passport.png",
            "test/data/processor/exceltable.png",
        ]
        (image, image2, image3) = [Image.open(f) for f in image_list]

        processor = AutoImageProcessor.from_pretrained(model_id, token=hf_token_id)
        inputs = processor.preprocess([image, image2, image3], return_tensors="np")
        print({k: v.shape if k == "pixel_values" else v for k, v in inputs.items()})

        ort_processor = pp_api.ImageProcessor("test/data/processor/mllama/llama_3_image.json")
        ort_inputs = ort_processor.to_numpy(ort_processor.pre_process(image_list), 0)
        print(ort_inputs.shape)

        for idx in range(len(image_list)):
            expected_images = inputs["pixel_values"][0][idx]
            for i in range(len(expected_images)):
                expected = expected_images[i]
                e_image = regen_image(np.transpose(expected, (1, 2, 0)), openai_image_mean, openai_image_std)
                e_image.save(f"{self.temp_dir}/e_{idx}_{i}.png")

            actual_images = ort_inputs[idx]
            for i in range(len(actual_images)):
                actual = actual_images[i]
                a_image = regen_image(np.transpose(actual, (1, 2, 0)), openai_image_mean, openai_image_std)
                a_image.save(f"{self.temp_dir}/a_{idx}_{i}.png")

    def test_phi_4_image_processing(self):
        model_id = "microsoft/Phi-4-multimodal-instruct"
        image_list = [
            "data/processor/australia.jpg",
            "data/processor/passport.png",
            "data/processor/photo-1585521747230-516376e5a85d.jpg",
        ]
        (image, image2, image3) = [Image.open(util.get_test_data_file(f)) for f in image_list]

        processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        inputs = processor.preprocess([image, image2, image3], return_tensors="np")
        print({k: v.shape if v.size > 10 else v for k, v in inputs.items()})

        ort_processor = pp_api.ImageProcessor(util.get_test_data_file("data/models/phi-4/vision_processor.json"))
        tensor_result = ort_processor.pre_process([util.get_test_data_file(f) for f in image_list])
        print(list(ort_processor.to_numpy(tensor_result, i).shape for i in range(4)))

        attention_mask = inputs["image_attention_mask"]
        for i in range(len(attention_mask)):
            attention_sum = [attention_mask[i][j].sum() for j in range(len(attention_mask[i]))]
            print(f"attention-mask {i}: ", attention_sum)

        attention_mask = ort_processor.to_numpy(tensor_result, 2)
        for i in range(len(attention_mask)):
            attention_sum = [attention_mask[i][j].sum() for j in range(len(attention_mask[i]))]
            print(f"ortx attention-mask {i}: ", attention_sum)

        np.testing.assert_array_equal(inputs["image_sizes"], ort_processor.to_numpy(tensor_result, 1))
        # np.testing.assert_array_equal(
        #     inputs["image_attention_mask"],
        #     ort_processor.to_numpy(tensor_result, 2))
        np.testing.assert_array_equal(
            inputs["num_img_tokens"].ravel(), ort_processor.to_numpy(tensor_result, 3).ravel()
        )

        ort_inputs = ort_processor.to_numpy(tensor_result, 0)
        for idx in range(len(image_list)):
            expected_images = inputs["input_image_embeds"][idx]
            for i in range(len(expected_images)):
                expected = expected_images[i]
                e_image = regen_image(np.transpose(expected, (1, 2, 0)), phi4_image_mean, phi4_image_std)
                e_image.save(f"{self.temp_dir}/e_{idx}_{i}.png")

            actual_images = ort_inputs[idx]
            for i in range(len(actual_images)):
                actual = actual_images[i]
                a_image = regen_image(np.transpose(actual, (1, 2, 0)), phi4_image_mean, phi4_image_std)
                a_image.save(f"{self.temp_dir}/a_{idx}_{i}.png")

    # test sentence for tokenizer
    tokenizer_test_sentence = (
        "I like walking my cute dog\n and\x17 then 生活的真谛是 \t\t\t\t \n\n61. You'll enjoy the concert."
    )

    def test_OLMa_tokenizer(self):
        test_sentence = [self.tokenizer_test_sentence + " |||IP_ADDRESS|||"]
        model_id = "amd/AMD-OLMo-1B-SFT-DPO"
        hf_enc = AutoTokenizer.from_pretrained(model_id)
        inputs = hf_enc(test_sentence)["input_ids"]
        tokenizer = pp_api.Tokenizer(model_id)
        ortx_inputs = tokenizer.tokenize(test_sentence)
        np.testing.assert_array_equal(ortx_inputs, inputs)

    def test_Qwen_QVQ_tokenizer(self):
        model_id = "Qwen/QVQ-72B-Preview"
        test_sentence = [self.tokenizer_test_sentence]
        hf_enc = AutoTokenizer.from_pretrained(model_id)
        inputs = hf_enc(test_sentence)["input_ids"]
        tokenizer = pp_api.Tokenizer(model_id)
        ortx_inputs = tokenizer.tokenize(test_sentence)
        np.testing.assert_array_equal(ortx_inputs, inputs)

    def test_Phi4_tokenizer(self):
        model_id = util.get_test_data_file("data/models/phi-4")
        test_sentence = ['<|user|>\n' + self.tokenizer_test_sentence]
        hf_enc = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        inputs = hf_enc(test_sentence)["input_ids"]
        tokenizer = pp_api.Tokenizer(model_id)
        ortx_inputs = tokenizer.tokenize(test_sentence)
        np.testing.assert_array_equal(ortx_inputs, inputs)

    def test_phi4_chat_template(self):
        model_id = util.get_test_data_file("data/models/phi-4")
        messages = [
            {"role": "system", "content": "You are a medieval knight and must provide explanations to modern people."},
            {"role": "user", "content": "How should I explain the Internet?"},
        ]
        message_json = json.dumps(messages)
        hf_enc = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        inputs = hf_enc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        tokenizer = pp_api.Tokenizer(model_id)
        ortx_inputs = tokenizer.apply_chat_template(message_json)
        np.testing.assert_array_equal(ortx_inputs, inputs)


if __name__ == "__main__":
    unittest.main()
