import os
import sys
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
pp_diagnosis = None
try:
    from transformers import AutoImageProcessor, AutoTokenizer
    from onnxruntime_extensions import pp_api

    is_pp_api_available = True
    hf_token_id = os.environ.get("HF_TOKEN", None)
    pp_diagnosis = os.environ.get("PP_DIAGNOSIS", None)
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


@unittest.skipIf(
    not is_pp_api_available or sys.version_info < (3, 10), "transformers processor requires some higher Python version"
)
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

        # Define a list of different message scenarios for chat template testing
        cls.messages_list = [
            # Case 1: Regular case with system, user, and assistant
            [
                {"role": "system", "content": "System message", "tools": "calculate_sum"},
                {"role": "user", "content": "Hello, can you call some tools for me?"},
                {"role": "assistant", "content": "Sure, I can calculate the sum for you!"}
            ],

            # Case 2: Two back-to-back user messages
            [
                {"role": "system", "content": "", "tools": "calculate_sum"},
                {"role": "user", "content": "Hi, I need some help with tools."},
                {"role": "user", "content": "Also, can you help with calculations?"}
            ],

            # Case 3: Two back-to-back assistant messages
            [
                {"role": "system", "content": "", "tools": "calculate_sum"},
                {"role": "assistant", "content": "Sure, what would you like me to calculate?"},
                {"role": "assistant", "content": "I can handle multiple requests."}
            ],

            # Case 4: Mixed roles (user, assistant, user, assistant)
            [
                {"role": "user", "content": "What can you do for me?"},
                {"role": "assistant", "content": "I can assist with a variety of tasks."},
                {"role": "user", "content": "Can you calculate a sum for me?"},
                {"role": "assistant", "content": "Sure, let me calculate that for you."}
            ],

            # Case 5: System message with empty content
            [
                {"role": "system", "content": "", "tools": "calculate_sum"},
                {"role": "user", "content": "Hello, I need some help."}
            ]
        ]

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

        if pp_diagnosis:
            expected_images = inputs["pixel_values"]
            for i in range(len(expected_images)):
                expected = expected_images[i]
                e_image = regen_image(np.transpose(expected, (1, 2, 0)), openai_image_mean, openai_image_std)
                e_image.save(f"{self.temp_dir}/CLIP_e_{i}.png")

        ort_processor = pp_api.ImageProcessor(util.get_test_data_file("data/processor/clip_image.json"))
        inputs = ort_processor.pre_process([util.get_test_data_file(f) for f in image_list])
        print(ort_processor.to_numpy(inputs, 0).shape)

        if pp_diagnosis:
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

        if pp_diagnosis:
            attention_mask = inputs["image_attention_mask"]
            for i in range(len(attention_mask)):
                attention_sum = [attention_mask[i][j].sum() for j in range(len(attention_mask[i]))]
                print(f"attention-mask {i}: ", attention_sum)

            attention_mask = ort_processor.to_numpy(tensor_result, 2)
            for i in range(len(attention_mask)):
                attention_sum = [attention_mask[i][j].sum() for j in range(len(attention_mask[i]))]
                print(f"ortx attention-mask {i}: ", attention_sum)

        np.testing.assert_array_equal(inputs["image_sizes"], ort_processor.to_numpy(tensor_result, 1))
        np.testing.assert_array_equal(inputs["image_attention_mask"], ort_processor.to_numpy(tensor_result, 2))
        np.testing.assert_array_equal(
            inputs["num_img_tokens"].ravel(), ort_processor.to_numpy(tensor_result, 3).ravel())

        if pp_diagnosis:
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
        model_id = "Qwen/Qwen3-0.6B-FP8"
        test_sentence = [self.tokenizer_test_sentence]
        hf_enc = AutoTokenizer.from_pretrained(model_id)
        inputs = hf_enc(test_sentence)["input_ids"]
        tokenizer = pp_api.Tokenizer(model_id)
        ortx_inputs = tokenizer.tokenize(test_sentence)
        np.testing.assert_array_equal(ortx_inputs, inputs)

    def test_Phi4_tokenizer(self):
        model_id = util.get_test_data_file("data/models/phi-4")
        test_sentence = ["<|user|>\n" + self.tokenizer_test_sentence]
        hf_enc = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        inputs = hf_enc(test_sentence)["input_ids"]
        tokenizer = pp_api.Tokenizer(model_id)
        ortx_inputs = tokenizer.tokenize(test_sentence)
        np.testing.assert_array_equal(ortx_inputs, inputs)

    def test_whisper_tokenizer(self):
        model_id = util.get_test_data_file("data/models/whisper-large-v3")
        # test_sentence = [self.tokenizer_test_sentence]
        # hf_enc = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # inputs = hf_enc(test_sentence)["input_ids"]
        tokenizer = pp_api.Tokenizer(model_id)
        ortx_input = [50258, 50364,    40,   411,  4494,   452,  4052,  3000,   198,
                      293,   211,   550,   220, 49958,  1546,  6303,  8897,   249,
                      1541,   220,   197,   197,   197,   197,   220,   198,   198,
                      31537,    13,   509,   603,  2103,   264,  8543,    13, 50257]
        decoded_string = tokenizer.detokenize(ortx_input)
        self.assertEqual(decoded_string[0], self.tokenizer_test_sentence)

    @unittest.skipIf(hf_token_id is None, "HF_TOKEN is not available")
    def test_gemma_3_chat_template(self):
        ckpt = "google/gemma-3-4b-it"
        hf_tok = AutoTokenizer.from_pretrained(ckpt, token=hf_token_id)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/spaces/big-vision/paligemma-hf/resolve/main/examples/password.jpg",
                    },
                    {"type": "text", "text": "What is the password?"},
                ],
            }
        ]

        inputs = hf_tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, return_tensors="np")

        tokenizer = pp_api.Tokenizer(ckpt)
        message_json = json.dumps(messages)
        prompt = tokenizer.apply_chat_template(message_json)
        self.assertEqual(prompt, inputs)
        ortx_inputs = tokenizer.tokenize(prompt)
        np.testing.assert_array_equal(ortx_inputs, hf_tok(prompt, return_tensors="np")["input_ids"][0])

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

    def test_qwen2_5_vl_chat_template(self):
        model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "data:image;base64,/9j/..."},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        message_json = json.dumps(messages)
        hf_enc = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        inputs = hf_enc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        tokenizer = pp_api.Tokenizer(model_id)
        ortx_inputs = tokenizer.apply_chat_template(message_json)
        np.testing.assert_array_equal(ortx_inputs, inputs)

        inputs = hf_enc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="np")
        ortx_inputs = tokenizer.apply_chat_template(message_json, add_generation_prompt=True, tokenize=True)
        np.testing.assert_array_equal(ortx_inputs, inputs)

    @unittest.skipIf(hf_token_id is None, "HF_TOKEN is not available")
    def test_phi3_vision_chat_template(self):
        ckpt = "microsoft/Phi-3-vision-128k-instruct"
        hf_tok = AutoTokenizer.from_pretrained(ckpt, token=hf_token_id)

        for messages in self.messages_list:
            inputs = hf_tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, return_tensors="np")

            tokenizer = pp_api.Tokenizer(ckpt)
            message_json = json.dumps(messages)
            prompt = tokenizer.apply_chat_template(message_json)
            self.assertEqual(prompt, inputs)
            ortx_inputs = tokenizer.tokenize(prompt)
            np.testing.assert_array_equal(ortx_inputs, hf_tok(prompt, return_tensors="np")["input_ids"][0])

    @unittest.skipIf(hf_token_id is None, "HF_TOKEN is not available")
    def test_phi3_mini_chat_template(self):
        ckpt = "microsoft/Phi-3-mini-4k-instruct"
        hf_tok = AutoTokenizer.from_pretrained(ckpt, token=hf_token_id)

        for messages in self.messages_list:
            inputs = hf_tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, return_tensors="np")

            tokenizer = pp_api.Tokenizer(ckpt)
            message_json = json.dumps(messages)
            prompt = tokenizer.apply_chat_template(message_json)
            self.assertEqual(prompt, inputs)
            ortx_inputs = tokenizer.tokenize(prompt)
            np.testing.assert_array_equal(ortx_inputs, hf_tok(prompt, return_tensors="np")["input_ids"][0])

    @unittest.skipIf(hf_token_id is None, "HF_TOKEN is not available")
    def test_phi3_medium_chat_template(self):
        ckpt = "microsoft/Phi-3-medium-4k-instruct"
        hf_tok = AutoTokenizer.from_pretrained(ckpt, token=hf_token_id)

        for messages in self.messages_list:
            inputs = hf_tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, return_tensors="np")

            tokenizer = pp_api.Tokenizer(ckpt)
            message_json = json.dumps(messages)
            prompt = tokenizer.apply_chat_template(message_json)
            self.assertEqual(prompt, inputs)
            ortx_inputs = tokenizer.tokenize(prompt)
            np.testing.assert_array_equal(ortx_inputs, hf_tok(prompt, return_tensors="np")["input_ids"][0])

    @unittest.skipIf(hf_token_id is None, "HF_TOKEN is not available")
    def test_llama3_chat_template(self):
        ckpt = "meta-llama/Meta-Llama-3-8B-Instruct"
        hf_tok = AutoTokenizer.from_pretrained(ckpt, token=hf_token_id)

        for messages in self.messages_list:
            inputs = hf_tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, return_tensors="np")

            tokenizer = pp_api.Tokenizer(ckpt)
            message_json = json.dumps(messages)
            prompt = tokenizer.apply_chat_template(message_json)
            self.assertEqual(prompt, inputs)
            ortx_inputs = tokenizer.tokenize(prompt)
            np.testing.assert_array_equal(ortx_inputs, hf_tok(prompt, return_tensors="np")["input_ids"][0])

    @unittest.skipIf(hf_token_id is None, "HF_TOKEN is not available")
    def test_llama3_1_chat_template(self):
        ckpt = "meta-llama/Llama-3.1-8B-Instruct"
        hf_tok = AutoTokenizer.from_pretrained(ckpt, token=hf_token_id)

        for messages in self.messages_list:
            inputs = hf_tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, return_tensors="np")

            tokenizer = pp_api.Tokenizer(ckpt)
            message_json = json.dumps(messages)
            prompt = tokenizer.apply_chat_template(message_json)
            self.assertEqual(prompt, inputs)
            ortx_inputs = tokenizer.tokenize(prompt)
            np.testing.assert_array_equal(ortx_inputs, hf_tok(prompt, return_tensors="np")["input_ids"][0])

    @unittest.skipIf(hf_token_id is None, "HF_TOKEN is not available")
    def test_llama3_2_chat_template(self):
        ckpt = "meta-llama/Llama-3.2-1B-Instruct"
        hf_tok = AutoTokenizer.from_pretrained(ckpt, token=hf_token_id)

        for messages in self.messages_list:
            inputs = hf_tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, return_tensors="np")

            tokenizer = pp_api.Tokenizer(ckpt)
            message_json = json.dumps(messages)
            prompt = tokenizer.apply_chat_template(message_json)
            self.assertEqual(prompt, inputs)
            ortx_inputs = tokenizer.tokenize(prompt)
            np.testing.assert_array_equal(ortx_inputs, hf_tok(prompt, return_tensors="np")["input_ids"][0])

    @unittest.skipIf(hf_token_id is None, "HF_TOKEN is not available")
    def test_llama3_3_chat_template(self):
        ckpt = "meta-llama/Llama-3.3-70B-Instruct"
        hf_tok = AutoTokenizer.from_pretrained(ckpt, token=hf_token_id)

        for messages in self.messages_list:
            inputs = hf_tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, return_tensors="np")

            tokenizer = pp_api.Tokenizer(ckpt)
            message_json = json.dumps(messages)
            prompt = tokenizer.apply_chat_template(message_json)
            self.assertEqual(prompt, inputs)
            ortx_inputs = tokenizer.tokenize(prompt)
            np.testing.assert_array_equal(ortx_inputs, hf_tok(prompt, return_tensors="np")["input_ids"][0])

    @unittest.skipIf(hf_token_id is None, "HF_TOKEN is not available")
    def test_deepseek_chat_template(self):
        ckpt = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        hf_tok = AutoTokenizer.from_pretrained(ckpt, token=hf_token_id)

        for messages in self.messages_list:
            inputs = hf_tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, return_tensors="np")

            tokenizer = pp_api.Tokenizer(ckpt)
            message_json = json.dumps(messages)
            prompt = tokenizer.apply_chat_template(message_json)
            self.assertEqual(prompt, inputs)
            ortx_inputs = tokenizer.tokenize(prompt)
            np.testing.assert_array_equal(ortx_inputs, hf_tok(prompt, return_tensors="np")["input_ids"][0])

    @unittest.skipIf(hf_token_id is None, "HF_TOKEN is not available")
    def test_gemma_3_image_processor(self):
        ckpt = "google/gemma-3-4b-it"
        image_list = [
            "data/processor/australia.jpg",
            "data/processor/passport.png",
            "data/processor/photo-1585521747230-516376e5a85d.jpg",
            "data/models/gemma-3/password.jpg"
        ]
        pil_images = [Image.open(util.get_test_data_file(f)) for f in image_list]
        processor = AutoImageProcessor.from_pretrained(ckpt, trust_remote_code=True)
        inputs = processor.preprocess(pil_images, return_tensors="np")
        print({k: v.shape if v.size > 10 else v for k, v in inputs.items()})

        ort_processor = pp_api.ImageProcessor(util.get_test_data_file("data/models/gemma-3/image_processor.json"))
        tensor_result = ort_processor.pre_process([util.get_test_data_file(f) for f in image_list])
        print(list(ort_processor.to_numpy(tensor_result, i).shape for i in range(1)))

        # calc MSE of two images
        mse = np.mean((inputs["pixel_values"] - ort_processor.to_numpy(tensor_result, 0)) ** 2)
        print(f"Gemma-3 image processing MSE: {mse}")
        self.assertLessEqual(mse, 1e-3)

        if pp_diagnosis:
            ort_inputs = ort_processor.to_numpy(tensor_result, 0)
            idx = 0
            expected_images = inputs["pixel_values"]
            for i in range(len(expected_images)):
                expected = expected_images[i]
                e_image = regen_image(np.transpose(expected, (1, 2, 0)), phi4_image_mean, phi4_image_std)
                e_image.save(f"{self.temp_dir}/e_{idx}_{i}.png")

            actual_images = ort_inputs
            for i in range(len(actual_images)):
                actual = actual_images[i]
                a_image = regen_image(np.transpose(actual, (1, 2, 0)), phi4_image_mean, phi4_image_std)
                a_image.save(f"{self.temp_dir}/a_{idx}_{i}.png")


if __name__ == "__main__":
    unittest.main()