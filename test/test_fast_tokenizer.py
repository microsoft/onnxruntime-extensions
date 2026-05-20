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


    def test_sequence_of_splits_pre_tokenizer(self):
        # Regression test for a Sequence pre_tokenizer containing multiple Split
        # regexes (HuggingFace allows this; models like tencent/Hy-MT1.5 ship one
        # with three Splits). Prior to this fix, ort-extensions silently kept only
        # the LAST Split in the Sequence, so any text matched by an earlier pattern
        # was mis-tokenised. We build a tiny synthetic tokenizer here rather than
        # downloading a remote model so the test is deterministic and offline.
        import json as _json
        import tempfile

        from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        # Byte-level BPE vocab over the 256-char ByteLevel alphabet (gpt2-style
        # mapping of raw bytes to printable code points). No merges -> every
        # token is one byte, which makes the round-trip easy to reason about
        # while still exercising the pre-tokeniser path end-to-end.
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
        tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        # Sequence with two Isolated Splits: digit runs (1-3) and a GPT2-style
        # word/whitespace pattern. The pre-fix bug would have dropped the digit
        # rule and only kept the GPT2 one.
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(r"\p{N}{1,3}"), behavior="isolated"),
            pre_tokenizers.Split(
                Regex(
                    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
                ),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tok.decoder = decoders.ByteLevel()
        hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok)

        with tempfile.TemporaryDirectory() as tmpdir:
            # gen_processing_models reads the saved tokenizer.json on disk.
            hf_tok.save_pretrained(tmpdir)
            # Sanity-check that the saved tokenizer.json really does have a
            # Sequence with multiple Splits (otherwise this test would regress
            # to the single-Split case and not exercise the bug).
            with open(f"{tmpdir}/tokenizer.json", encoding="utf-8") as fh:
                spec = _json.load(fh)
            pretokens = spec["pre_tokenizer"]["pretokenizers"]
            splits = [p for p in pretokens if p.get("type") == "Split"]
            self.assertGreaterEqual(len(splits), 2)

            reloaded = AutoTokenizer.from_pretrained(tmpdir)
            ort_tok, _ = gen_processing_models(reloaded, pre_kwargs={})

        # Mix of words and digit runs to exercise both Split patterns.
        text = ["hello 123 world", "abc 42 def 9 ghi", "no digits here"]
        actual_ids, *_ = ort_inference(ort_tok, text)
        pad_id = reloaded.pad_token_id  # may be None for tokenizers without an explicit pad
        for n, t in enumerate(text):
            expected = reloaded.encode(t, return_tensors="np")[0]
            row = actual_ids[n]
            if pad_id is not None:
                row = row[row != pad_id]
            # Length must match exactly; a slice-then-compare would let extra
            # trailing tokens pass silently.
            self.assertEqual(
                len(row),
                len(expected),
                f"Token count mismatch for {t!r}: ort={list(row)} hf={list(expected)}",
            )
            np.testing.assert_array_equal(expected, row)


if __name__ == '__main__':
    unittest.main()
