from pathlib import Path
import unittest
import numpy as np
import transformers
from onnxruntime_extensions import PyOrtFunction, BertTokenizer


def _get_test_data_file(*sub_dirs):
    test_dir = Path(__file__).parent
    return str(test_dir.joinpath(*sub_dirs))


def _run_bert_tokenizer(input, vocab_path):
    t2stc = PyOrtFunction.from_customop(BertTokenizer, vocab_file=vocab_path)
    result = t2stc(input)
    print(result)


class TestBertTokenizer(unittest.TestCase):

    def test_text_to_case1(self):
        _run_bert_tokenizer(input="", vocab_path=_get_test_data_file('data', 'bert_basic_cased_vocab.txt'))


if __name__ == "__main__":
    unittest.main()
