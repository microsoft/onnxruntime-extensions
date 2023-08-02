# coding: utf-8
import unittest
import numpy as np
import transformers
from onnxruntime_extensions import PyOrtFunction, BertTokenizer, util
from transformers import BertTokenizerFast


bert_cased_tokenizer = transformers.BertTokenizer(
    util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
    False,
    strip_accents=True,
)


def _run_basic_case(input, vocab_path):
    t2stc = PyOrtFunction.from_customop(
        BertTokenizer, vocab_file=vocab_path, do_lower_case=0, strip_accents=1
    )
    result = t2stc([input])
    expect_result = bert_cased_tokenizer.encode_plus(input)
    np.testing.assert_array_equal(result[0], expect_result["input_ids"])
    np.testing.assert_array_equal(result[1], expect_result["token_type_ids"])
    np.testing.assert_array_equal(result[2], expect_result["attention_mask"])


def _run_combined_case(input, vocab_path):
    t2stc = PyOrtFunction.from_customop(
        BertTokenizer, vocab_file=vocab_path, do_lower_case=0, strip_accents=1
    )
    result = t2stc(input)
    expect_result = bert_cased_tokenizer.encode_plus(input[0], input[1])
    np.testing.assert_array_equal(result[0], expect_result["input_ids"])
    np.testing.assert_array_equal(result[1], expect_result["token_type_ids"])
    np.testing.assert_array_equal(result[2], expect_result["attention_mask"])

def _run_basic_with_offset_check(input, vocab_path):
    t2stc = PyOrtFunction.from_customop(
        BertTokenizer, vocab_file=vocab_path, do_lower_case=0, strip_accents=1
    )
    result = t2stc([input])
    expect_result = bert_cased_tokenizer.encode_plus(input)
    np.testing.assert_array_equal(result[0], expect_result["input_ids"])
    np.testing.assert_array_equal(result[1], expect_result["token_type_ids"])
    np.testing.assert_array_equal(result[2], expect_result["attention_mask"])
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    bert_out = tokenizer(input, return_offsets_mapping=True)
    np.testing.assert_array_equal(result[3], bert_out['offset_mapping'])
    print("\nTest sentence: " + str(input))
    print("HF offset mapping: " + str(bert_out['offset_mapping']))
    print("EXT offset mapping: ", end='')
    for row in result[3]:
        print("(" + str(row[0]) + ", " + str(row[1]) + "), ", end='')
    print("\n")


class TestBertTokenizer(unittest.TestCase):
    def test_text_to_case1(self):

        print("\n\n****** Starting input ids, token type ids, and attention mask tests. ******\n")

        _run_basic_case(
            input="Input 'text' must not be empty.",
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )
        _run_basic_case(
            input="ÀÁÂÃÄÅÇÈÉÊËÌÍÎÑÒÓÔÕÖÚÜ\t"
            + "䗓𨖷虴𨀐辘𧄋脟𩑢𡗶镇伢𧎼䪱轚榶𢑌㺽𤨡!#$%&(Tom@microsoft.com)*+,-./:;<=>?@[\\]^_`{|}~",
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )
        _run_basic_case(
            input="网易云音乐",
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )
        _run_basic_case(
            input="本想好好的伤感　想放任　但是没泪痕",
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )
        _run_combined_case(
            ["网 易 云 音 乐", "cat isnot playing toyssss"],
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )
        print("\n****** Input ids, token type ids, and attention mask tests complete. ******\n\n\n")
        print("*** Starting offset mapping tests. ***\n")

        _run_basic_with_offset_check(
            input="网 易 云 音 乐",
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )
        _run_basic_with_offset_check(
            input="cat is playing toys",
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )
        _run_basic_with_offset_check(
            input="cat isnot playing toyssss",
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )
        _run_basic_with_offset_check(
            input="ah oui on peut parler francais",
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )
        _run_basic_with_offset_check(
            input="und eigentlich auch deutsch",
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )
        _run_basic_with_offset_check(
            input="podemos hablar muchos idiomas",
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )
        _run_basic_with_offset_check(
            input="力",
            vocab_path=util.get_test_data_file("data", "bert_basic_cased_vocab.txt"),
        )

        print("\n*** Offset mapping tests complete. ***\n")


if __name__ == "__main__":
    unittest.main()
