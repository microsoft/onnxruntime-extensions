# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
from transformers import AutoTokenizer
from onnxruntime_extensions import OrtPyFunction, gen_processing_models


class TestAutoTokenizer(unittest.TestCase):
    def test_bert_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors='np')
        ort_tok = OrtPyFunction(gen_processing_models(tokenizer, pre_kwargs={})[0])
        actual_ids = ort_tok([text])[0]
        np.testing.assert_array_equal(encoded_input['input_ids'][0], actual_ids)

    def test_llama_tokenizer(self):
        # replace the official model name after the model is not gated anymore
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        text = "I was born in 92000, and this is falsé."
        ids = tokenizer.encode(text, return_tensors="np")

        ort_tok = OrtPyFunction.from_model(gen_processing_models(
            tokenizer,
            pre_kwargs={"WITH_DEFAULT_INPUTS": True})[0])
        actual_ids = ort_tok([text])[0]
        np.testing.assert_array_equal(ids[0], actual_ids)

    def test_falcon_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b", use_fast=False)
        text = "why don't you teach me some German?"
        ids = tokenizer.encode(text, return_tensors="np")

        ort_tok = OrtPyFunction.from_model(gen_processing_models(
            tokenizer,
            pre_kwargs={"WITH_DEFAULT_INPUTS": True})[0])
        actual_ids = ort_tok([text])[0]
        np.testing.assert_array_equal(ids, actual_ids)

    def test_t5_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
        text = "best hotel in bay area."
        ids = tokenizer.encode(text, return_tensors="np")
        ort_tok = OrtPyFunction.from_model(gen_processing_models(tokenizer, pre_kwargs={})[0])
        actual_ids = ort_tok([text])[0]
        np.testing.assert_array_equal(ids[0], actual_ids)

    def test_roberta_base(self):
        tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions", use_fast=False)
        text = "Agree. Keep trying, then if your rejected every time. I'm sorry your done."
        ids = tokenizer.encode(text, return_tensors="np")
        m_tok, m_detok = gen_processing_models(tokenizer, pre_kwargs={}, post_kwargs={})

        actual_ids = OrtPyFunction(m_tok)([text])[0]
        np.testing.assert_array_equal(ids, actual_ids)

        self.assertEqual(OrtPyFunction(m_detok)(ids)[0], tokenizer.decode(ids[0]))

    def test_clip_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
        text = "Wow, these models are getting popular."
        ids = tokenizer.encode(text, return_tensors="np")

        ort_tok = OrtPyFunction.from_model(gen_processing_models(
            tokenizer,
            pre_kwargs={"WITH_DEFAULT_INPUTS": True})[0])
        actual_ids = ort_tok([text])[0]
        np.testing.assert_array_equal(ids, actual_ids)


if __name__ == '__main__':
    unittest.main()
