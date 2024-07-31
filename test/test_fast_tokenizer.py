# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
from transformers import AutoTokenizer
from onnxruntime_extensions import gen_processing_models, ort_inference


class TestAutoTokenizer(unittest.TestCase):
    def test_llama_tokenizer(self):
        # replace the official model name after the model is not gated anymore
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer", use_fast=True)
        text = "I was born in 92000, and this is falsé."
        ids = tokenizer.encode(text, return_tensors="np")

        ort_tok, _ = gen_processing_models(tokenizer, pre_kwargs={})
        actual_ids, *_ = ort_inference(ort_tok, [text])
        self.assertEqual(actual_ids.dtype, np.int64)
        np.testing.assert_array_equal(ids[0], actual_ids[0])

    def test_mistral(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "mistral-community/Mistral-7B-v0.2", use_fast=True)
        text = "\nOnce upon a time, I was really into monochromatic makeup looks. I have a lot of coppery and bronze "
        ids = tokenizer.encode(text, return_tensors="np")

        ort_tok, _ = gen_processing_models(tokenizer, pre_kwargs={})
        actual_ids, *_ = ort_inference(ort_tok, [text])
        np.testing.assert_array_equal(ids[0], actual_ids[0])

    def test_phi_3_mini(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct", use_fast=True)
        text = ["what are you? \n 给 weiss ich, über was los ist \n",
                "@? \n was los ist \n",
                "Qué dijiste? \n über 给 ば was los ist im Mannschaft ц \n",
                "明天雷阵雨，气温26度。"]

        ids = tokenizer(text, return_tensors="np")["input_ids"]

        ort_tok, _ = gen_processing_models(tokenizer, pre_kwargs={})
        actual_ids, *_ = ort_inference(ort_tok, text)
        for n in range(len(actual_ids)):
            print("index is ", n)
            expected_ids = tokenizer.encode(text[n], return_tensors="np")
            np.testing.assert_array_equal(expected_ids[0], actual_ids[n][1:])


if __name__ == '__main__':
     unittest.main()
