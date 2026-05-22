# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import os
import tempfile
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


    @staticmethod
    def _create_hunyuan_like_tokenizer(tokenizer_dir):
        """Build a minimal Hunyuan-style tokenizer with a Sequence of 3 Split steps + ByteLevel."""
        from tokenizers import Regex, Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split

        vocab = {token: idx for idx, token in enumerate(ByteLevel.alphabet())}
        vocab["<unk>"] = len(vocab)

        tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
        tokenizer.pre_tokenizer = Sequence([
            Split(Regex(r"\p{N}{1,3}"), behavior="isolated"),
            Split(Regex("[一-龥぀-ゟ゠-ヿ]+"), behavior="isolated"),
            Split(Regex(
                r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"""
                r"""|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"""
                r"""| ?[\p{P}\p{S}]+[\r\n]*"""
                r"""|\s*[\r\n]+"""
                r"""|\s+(?!\S)"""
                r"""|\s+"""
            ), behavior="isolated"),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer.save(tokenizer_json_path)

        with open(tokenizer_json_path, encoding="utf-8") as f:
            tokenizer_json = json.load(f)
        split_regex = tokenizer_json["pre_tokenizer"]["pretokenizers"][2]["pattern"]["Regex"]
        tokenizer_json["pre_tokenizer"]["pretokenizers"][2]["pattern"]["Regex"] = (
            split_regex.replace(r"\r", "\r").replace(r"\n", "\n")
        )
        with open(tokenizer_json_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, ensure_ascii=False)

        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "tokenizer_class": "PreTrainedTokenizerFast",
                "add_bos_token": False,
                "add_eos_token": False,
            }, f)

        return tokenizer

    def test_hunyuan_split_sequence_tokenization(self):
        """Verify that a Sequence of multiple Split steps + ByteLevel produces
        the same token IDs as HuggingFace's tokenizers library."""
        with tempfile.TemporaryDirectory() as tokenizer_dir:
            tokenizer = self._create_hunyuan_like_tokenizer(tokenizer_dir)
            ort_tok, _ = gen_processing_models(tokenizer_dir, pre_kwargs={}, schema_v2=True)

            text = [
                "abc1234中文!Hello",
                "Hello\tWorld",
                "Line1\nLine2",
                "价格是￥12.50",
            ]

            ort_outputs = ort_inference(ort_tok, text)
            actual_ids = ort_outputs[0] if isinstance(ort_outputs, tuple) else ort_outputs
            for n, sample in enumerate(text):
                expected_ids = np.asarray(tokenizer.encode(sample).ids, dtype=np.int64)
                try:
                    np.testing.assert_array_equal(expected_ids, actual_ids[n][:expected_ids.shape[0]])
                except AssertionError:
                    print("the failed Hunyuan sentence index is ", n)
                    print("the failed Hunyuan sentence is ", sample.encode("unicode_escape").decode("ascii"))
                    raise


    def test_split_sequence_with_fallback_regex(self):
        """Verify that SplitIsolated works when a Split regex is NOT in the
        hardcoded pattern table, forcing the STL std::regex fallback path."""
        from tokenizers import Regex, Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split

        with tempfile.TemporaryDirectory() as tokenizer_dir:
            vocab = {token: idx for idx, token in enumerate(ByteLevel.alphabet())}
            vocab["<unk>"] = len(vocab)

            tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
            tokenizer.pre_tokenizer = Sequence([
                # [aeiou]+ is not in the hardcoded pattern table — forces fallback
                Split(Regex(r"[aeiou]+"), behavior="isolated"),
                ByteLevel(add_prefix_space=False, use_regex=False),
            ])
            tokenizer.decoder = ByteLevelDecoder()

            tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
            tokenizer.save(tokenizer_json_path)
            with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "add_bos_token": False,
                    "add_eos_token": False,
                }, f)

            ort_tok, _ = gen_processing_models(tokenizer_dir, pre_kwargs={}, schema_v2=True)

            text = ["hello world", "aeiou", "xyz", "banana"]

            ort_outputs = ort_inference(ort_tok, text)
            actual_ids = ort_outputs[0] if isinstance(ort_outputs, tuple) else ort_outputs
            for n, sample in enumerate(text):
                expected_ids = np.asarray(tokenizer.encode(sample).ids, dtype=np.int64)
                try:
                    np.testing.assert_array_equal(expected_ids, actual_ids[n][:expected_ids.shape[0]])
                except AssertionError:
                    print("the failed fallback-regex sentence index is ", n)
                    print("the failed fallback-regex sentence is ", repr(sample))
                    raise


if __name__ == '__main__':
    unittest.main()
