# coding: utf-8
import unittest
import numpy as np
import transformers
from onnxruntime_extensions import OrtPyFunction, pnp, util


_vocab_file = util.get_test_data_file("data", "bert_basic_cased_vocab.txt")

_bert_cased_tokenizer = transformers.BertTokenizer(
    _vocab_file, False, strip_accents=True)


def _run_basic_case(t2stc, input):
    actual_result = t2stc([input])
    expect_result = _bert_cased_tokenizer.encode_plus(input)
    np.testing.assert_array_equal(actual_result[0], expect_result["input_ids"])
    np.testing.assert_array_equal(actual_result[1], expect_result["token_type_ids"])
    np.testing.assert_array_equal(actual_result[2], expect_result["attention_mask"])


class TestBertTokenizerOp(unittest.TestCase):
    def test_text_to_case1_with_vocab_file(self):
        ort_tok = pnp.PreHuggingFaceBert(vocab_file=_vocab_file, do_lower_case=0, strip_accents=1)
        model = pnp.export(pnp.SequentialProcessingModule(ort_tok), ["whatever"])
        t2stc = OrtPyFunction.from_model(model)

        _run_basic_case(
            t2stc,
            input="Input 'text' must not be empty."
        )
        _run_basic_case(
            t2stc,
            input="ÀÁÂÃÄÅÇÈÉÊËÌÍÎÑÒÓÔÕÖÚÜ\t"
            + "䗓𨖷虴𨀐辘𧄋脟𩑢𡗶镇伢𧎼䪱轚榶𢑌㺽𤨡!#$%&(Tom@microsoft.com)*+,-./:;<=>?@[\\]^_`{|}~"
        )
        _run_basic_case(
            t2stc,
            input="网易云音乐"
        )
        _run_basic_case(
            t2stc,
            input="本想好好的伤感　想放任　但是没泪痕"
        )
        _run_basic_case(
            t2stc,
            input="网 易 云 音 乐"
        )
        _run_basic_case(
            t2stc,
            input="cat is playing toys"
        )
        _run_basic_case(
            t2stc,
            input="cat isnot playing toyssss"
        )


    def test_text_to_case1_with_hf_tok(self):
        ort_tok = pnp.PreHuggingFaceBert(hf_tok=_bert_cased_tokenizer)
        model = pnp.export(pnp.SequentialProcessingModule(ort_tok), ["whatever"])
        t2stc = OrtPyFunction.from_model(model)

        _run_basic_case(
            t2stc,
            input="Input 'text' must not be empty."
        )
        _run_basic_case(
            t2stc,
            input="ÀÁÂÃÄÅÇÈÉÊËÌÍÎÑÒÓÔÕÖÚÜ\t"
            + "䗓𨖷虴𨀐辘𧄋脟𩑢𡗶镇伢𧎼䪱轚榶𢑌㺽𤨡!#$%&(Tom@microsoft.com)*+,-./:;<=>?@[\\]^_`{|}~"
        )
        _run_basic_case(
            t2stc,
            input="网易云音乐"
        )
        _run_basic_case(
            t2stc,
            input="本想好好的伤感　想放任　但是没泪痕"
        )
        _run_basic_case(
            t2stc,
            input="网 易 云 音 乐"
        )
        _run_basic_case(
            t2stc,
            input="cat is playing toys"
        )
        _run_basic_case(
            t2stc,
            input="cat isnot playing toyssss"
        )


if __name__ == "__main__":
    unittest.main()
