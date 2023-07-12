import unittest
import numpy as np

from transformers import AutoProcessor
from onnxruntime_extensions import PyOrtFunction
from onnxruntime_extensions.cvt import HFTokenizerConverter

input("puase")

class TestBpeTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.hf_processor = AutoProcessor.from_pretrained(
            "openai/whisper-tiny.en")
        cls.tokenizer_cvt = HFTokenizerConverter(cls.hf_processor.tokenizer)
        return super().setUpClass()

    def test_bpe_tokenizer(self):
        fn_tokenizer = PyOrtFunction.from_customop(
            "GPT2Tokenizer",
            cvt=(self.tokenizer_cvt).bpe_tokenizer)
        test_str = " Lennils, pictures are a sort of upguards and atom paintings, and Mason's exquisite idles"
        test_ids = self.hf_processor.tokenizer.encode(test_str)
        self.assertTrue(fn_tokenizer(test_ids), test_str)

    def test_en_decoder(self):
        special_tokens = False
        fn_decoder = PyOrtFunction.from_customop(
            "BpeDecoder",
            cvt=self.tokenizer_cvt.bpe_decoder,
            skip_special_tokens=not special_tokens)
        test_str = "Hey! How are you feeling? J'ai l'impression que 郷さん est prêt"
        test_token_ids = self.hf_processor.tokenizer.encode(test_str)
        expected_str = self.hf_processor.tokenizer.decode(
            test_token_ids, skip_special_tokens=not special_tokens)
        self.assertEqual(fn_decoder(np.asarray(test_token_ids)), expected_str)

    def test_en_decoder_with_special(self):
        special_tokens = True
        fn_decoder = PyOrtFunction.from_customop(
            "BpeDecoder",
            cvt=self.tokenizer_cvt.bpe_decoder,
            skip_special_tokens=not special_tokens)
        test_str = "Hey! How are you feeling? J'ai l'impression que 郷さん est prêt"
        test_token_ids = self.hf_processor.tokenizer.encode(test_str)
        expected_str = self.hf_processor.tokenizer.decode(
            test_token_ids, skip_special_tokens=not special_tokens)
        actual_str = fn_decoder(np.asarray(test_token_ids))
        self.assertEqual(actual_str[0], expected_str)


if __name__ == "__main__":
    unittest.main()
