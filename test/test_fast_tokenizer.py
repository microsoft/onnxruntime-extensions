# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import transformers
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

    @unittest.skipIf(transformers.__version__ > "4.50",
                     reason="mistral tokenizer protobuf is out of date")
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
            "microsoft/Phi-3-mini-128k-instruct", use_fast=True, add_bos_token=True, add_eos_token=False)
        text = ["what are you? \n 给 weiss ich, über was los ist \n",
                "@? \n was los ist \n",
                "Qué dijiste? \n über 给 ば was los ist im Mannschaft ц \n",
                "明天雷阵雨，气温26度。"]

        ort_tok, _ = gen_processing_models(tokenizer, pre_kwargs={})
        actual_ids, *_ = ort_inference(ort_tok, text)
        for n in range(len(actual_ids)):
            expected_ids = tokenizer.encode(text[n], return_tensors="np")
            try:
                np.testing.assert_array_equal(
                    # skip the padding tokens in the ort output
                    expected_ids[0], actual_ids[n][:expected_ids.shape[1]])
            except AssertionError:
                print("the failed sentence index is ", n)
                raise


    def test_hy_mt_sequence_of_splits(self):
        # tencent/Hy-MT1.5-1.8B-2bit has a pre_tokenizer of type Sequence containing
        # three Split regexes (digits-of-1-to-3, CJK runs, then a GPT2-style fallback).
        # Prior to honouring every Split in the Sequence, only the last regex was kept,
        # which mistokenised plain English as a single byte token.
        tokenizer = AutoTokenizer.from_pretrained(
            "tencent/Hy-MT1.5-1.8B-2bit", use_fast=True, trust_remote_code=True)
        text = ["Hello, world!", "The cat is sleeping.", "明天有雨。"]

        ort_tok, _ = gen_processing_models(tokenizer, pre_kwargs={})
        actual_ids, *_ = ort_inference(ort_tok, text)
        for n, t in enumerate(text):
            expected_ids = tokenizer.encode(t, return_tensors="np")
            np.testing.assert_array_equal(
                expected_ids[0], actual_ids[n][:expected_ids.shape[1]])


if __name__ == '__main__':
    unittest.main()
