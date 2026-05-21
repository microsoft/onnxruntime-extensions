# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import tempfile
import unittest
import transformers
import numpy as np
from transformers import AutoTokenizer
from onnxruntime_extensions import gen_processing_models, ort_inference


HUNYUAN_SPLIT_REGEX = (
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"""
    r"""|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"""
    r"""| ?[\p{P}\p{S}]+[\r\n]*"""
    r"""|\s*[\r\n]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


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
            Split(Regex(HUNYUAN_SPLIT_REGEX), behavior="isolated"),
            ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer_json_path = f"{tokenizer_dir}/tokenizer.json"
        tokenizer.save(tokenizer_json_path)

        with open(tokenizer_json_path, encoding="utf-8") as f:
            tokenizer_json = json.load(f)
        split_regex = tokenizer_json["pre_tokenizer"]["pretokenizers"][2]["pattern"]["Regex"]
        # Match real tokenizer.json files where JSON parsing decodes CR/LF regex escapes to control bytes.
        tokenizer_json["pre_tokenizer"]["pretokenizers"][2]["pattern"]["Regex"] = (
            split_regex.replace(r"\r", "\r").replace(r"\n", "\n")
        )
        with open(tokenizer_json_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, ensure_ascii=False)

        with open(f"{tokenizer_dir}/tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "tokenizer_class": "PreTrainedTokenizerFast",
                "add_bos_token": False,
                "add_eos_token": False,
            }, f)

        return tokenizer

    def test_hunyuan_split_sequence_pretokenizer_shape(self):
        with tempfile.TemporaryDirectory() as tokenizer_dir:
            self._create_hunyuan_like_tokenizer(tokenizer_dir)
            with open(f"{tokenizer_dir}/tokenizer.json", encoding="utf-8") as f:
                tokenizer_json = json.load(f)

        pre_tokenizer = tokenizer_json["pre_tokenizer"]
        self.assertEqual(pre_tokenizer["type"], "Sequence")

        pretokenizers = pre_tokenizer["pretokenizers"]
        self.assertEqual([p["type"] for p in pretokenizers], ["Split", "Split", "Split", "ByteLevel"])

        split_regexes = [p["pattern"]["Regex"] for p in pretokenizers if p["type"] == "Split"]
        self.assertEqual(split_regexes[0], r"\p{N}{1,3}")
        self.assertEqual(split_regexes[1], "[一-龥぀-ゟ゠-ヿ]+")
        self.assertIn("[A-Za-z]+", split_regexes[2])
        self.assertIn("\r", split_regexes[2])
        self.assertIn("\n", split_regexes[2])

    def test_hunyuan_split_sequence_tokenization(self):
        with tempfile.TemporaryDirectory() as tokenizer_dir:
            tokenizer = self._create_hunyuan_like_tokenizer(tokenizer_dir)
            ort_tok, _ = gen_processing_models(tokenizer_dir, pre_kwargs={}, schema_v2=True)

            text = [
                "abc1234中文!Hello",
                "龥鿿",
                "Hello\tWorld",
                "Line1\nLine2",
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


if __name__ == '__main__':
    unittest.main()
