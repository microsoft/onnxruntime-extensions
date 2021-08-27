from pathlib import Path
import unittest
import numpy as np
import transformers
from onnxruntime_extensions import PyOrtFunction, BertTokenizerDecoder

bert_cased_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
bert_uncased_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

def _get_test_data_file(*sub_dirs):
    test_dir = Path(__file__).parent
    return str(test_dir.joinpath(*sub_dirs))


def _run_basic_case(input, vocab_path):
    t2stc = PyOrtFunction.from_customop(BertTokenizerDecoder, vocab_file=vocab_path)
    ids = np.array(bert_cased_tokenizer.encode(input), dtype=np.int64)
    position = np.array([[0, ids.size]], dtype=np.int64)

    result = t2stc(ids, position)
    np.testing.assert_array_equal(result[0], bert_cased_tokenizer.decode(bert_cased_tokenizer.encode(input), True, False))


class TestBertTokenizerDecoder(unittest.TestCase):

    def test_text_to_case1(self):
        _run_basic_case(input="Input 'text' must not be empty.",
                        vocab_path=_get_test_data_file('data', 'bert_basic_cased_vocab.txt'))
        _run_basic_case(input="网易云音乐", vocab_path=_get_test_data_file('data', 'bert_basic_cased_vocab.txt'))
        _run_basic_case(input="网 易 云 音 乐",
                        vocab_path=_get_test_data_file('data', 'bert_basic_cased_vocab.txt'))
        _run_basic_case(input="cat is playing toys",
                        vocab_path=_get_test_data_file('data', 'bert_basic_cased_vocab.txt'))
        _run_basic_case(input="cat isnot playing toyssss",
                        vocab_path=_get_test_data_file('data', 'bert_basic_cased_vocab.txt'))



if __name__ == "__main__":
    unittest.main()
