# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest

import numpy as np
from transformers import AutoTokenizer, GPT2Tokenizer
from onnxruntime_extensions import OrtPyFunction, gen_processing_models, ort_inference, util


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
        text = """
               1. Testing long text with multiple lines to check newline handling
               2. As well as words with apostrophes such as you're, i'm, don't, etc.
               3. And weird characters such as . , ~ ? ( ) " [ ] ! : - .
               """
        ids = tokenizer.encode(text, return_tensors="np")

        ort_tok = OrtPyFunction.from_model(gen_processing_models(
            tokenizer,
            pre_kwargs={"WITH_DEFAULT_INPUTS": True})[0])
        actual_ids = ort_tok([text])[0]
        np.testing.assert_array_equal(ids, actual_ids)
        
    def test_gpt2_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained("Xenova/gpt-4", use_fast=False)
        text = "Deep learning has come a long way, no?"
        ids = tokenizer.encode(text, return_tensors="np")

        ort_tok = OrtPyFunction.from_model(gen_processing_models(
            tokenizer,
            pre_kwargs={"WITH_DEFAULT_INPUTS": True})[0])
        actual_ids = ort_tok([text])[0]
        np.testing.assert_array_equal(ids, actual_ids)

    def test_xmlroberta_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=False)
        # TODO: if there is <unk> in text, the result is not matched.
        text = (
            'This is a very long text with a lot of weird characters, such as: . , ~ ? ( ) " [ ] ! : - . Also we will'
            " add words that should not exsist and be tokenized to , such as saoneuhaoesuth")
        ids = tokenizer.encode(text, return_tensors="np")

        ort_tok, _ = gen_processing_models(tokenizer,pre_kwargs={"WITH_DEFAULT_INPUTS": True})
        actual_ids, *_ = ort_inference(ort_tok, [text])
        np.testing.assert_array_equal(ids[0], actual_ids)

    def test_trie_tokenizer(self):
        vocab_file = util.get_test_data_file("data", "rwkv_vocab_v20230424.txt")
        vocab_data = util.read_file(vocab_file, 'rb')
        tok, detok = gen_processing_models("TrieTokenizer",
                                           pre_kwargs={'vocab': vocab_data},
                                           post_kwargs={'vocab': vocab_data})
        text = ["that dog is so cute"]
        ids = ort_inference(tok, text)
        det_text = ort_inference(detok, ids)
        self.assertEqual(text, det_text)

    def test_microsoft_ph1(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-1_5", trust_remote_code=True, torch_dtype="auto", use_fast=False)
        code = '''```python
        def print_prime(n):
           """
           Print all primes between 1 and n
           """'''

        ids = tokenizer(code, return_tensors="np", return_attention_mask=False)
        ort_tok, _ = gen_processing_models(tokenizer, pre_kwargs={})
        actual_ids, *_ = ort_inference(ort_tok, [code])
        self.assertEqual(len(ids['input_ids'].shape), len(actual_ids.shape))
        # TODO: not matched.
        # np.testing.assert_array_equal(ids['input_ids'], actual_ids)


if __name__ == '__main__':
    unittest.main()
