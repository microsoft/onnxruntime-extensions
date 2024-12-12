# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
from transformers import AutoTokenizer, GPT2Tokenizer
from onnxruntime_extensions import OrtPyFunction, gen_processing_models, ort_inference


class TestEmbeddedTokenizer(unittest.TestCase):
    def test_clip_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32", use_fast=False)
        text = """
               1. Testing long text with multiple lines to check newline handling
               2. As well as words with apostrophes such as you're, i'm, don't, etc.
               3. And weird characters such as . , ~ ? ( ) " [ ] ! : - .
               """
        ids = tokenizer.encode(text, return_tensors="np")

        ort_tok = OrtPyFunction.from_model(gen_processing_models(
            tokenizer,
            pre_kwargs={"WITH_DEFAULT_INPUTS": True})[0],
            schema_v2=True)
        actual_ids = ort_tok([text])[0]
        np.testing.assert_array_equal(ids, actual_ids)

    def test_gpt2_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained(
            "Xenova/gpt-4", use_fast=False)
        text = "Testing words with apostrophes such as you're, i'm, don't, etc."
        ids = tokenizer.encode(text, return_tensors="np")

        ort_tok = OrtPyFunction.from_model(gen_processing_models(
            tokenizer,
            pre_kwargs={"WITH_DEFAULT_INPUTS": True})[0],
            schema_v2=True)
        actual_ids = ort_tok([text])[0]
        np.testing.assert_array_equal(ids, actual_ids)

    def test_xlm_roberta_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "xlm-roberta-base", use_fast=False)
        # TODO: if there is <unk> in text, the result is not matched.
        text = (
            'This is a very long text with a lot of weird characters, such as: . , ~ ? ( ) " [ ] ! : - . Also we will'
            " add words that should not exist and be tokenized to , such as saoneuhaoesuth")
        ids = tokenizer.encode(text, return_tensors="np")

        ort_tok, _ = gen_processing_models(
            tokenizer,
            pre_kwargs={"WITH_DEFAULT_INPUTS": True},
            schema_v2=True)
        actual_ids, *_ = ort_inference(ort_tok, [text])
        np.testing.assert_array_equal(ids[0], actual_ids)


if __name__ == '__main__':
    unittest.main()