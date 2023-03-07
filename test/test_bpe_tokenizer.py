import unittest
import numpy as np

from transformers import AutoProcessor
from onnxruntime_extensions import PyOrtFunction
from onnxruntime_extensions.cvt import HFTokenizerConverter


class TestBpeTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.hf_processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        cls.tokenizer_cvt = HFTokenizerConverter(cls.hf_processor.tokenizer)
        return super().setUpClass()
    
    @unittest.skip("Not implemented")
    def test_bpe_tokenizer(self):
        fn_tokenizer = PyOrtFunction.from_customop("GPT2Tokenizer",
                                        cvt=(self.tokenizer_cvt).bbpe_tokenizer)

        # fmt: off
        test_ids = [
            441, 1857, 4174, 11, 5242, 366,
            257, 1333, 295, 493, 2794, 2287, 293, 12018, 14880, 11,
            293, 25730, 311, 454, 34152, 4496, 904, 50724
        ]
        # fmt: on

        self.assertTrue(fn_tokenizer(test_ids),
                        " Lennils, pictures are a sort of upguards and atom paintings, and Mason's exquisite idles")

    def test_en_decoder(self):
        fn_decoder = PyOrtFunction.from_customop("BpeDecoder",
                                        cvt=self.tokenizer_cvt.bpe_decoder,
                                        skip_special_tokens=False)
        test_str = "Hey! How are you feeling? J'ai l'impression que 郷さん est prêt"
        # test_token_ids = self.hf_processor.tokenizer.encode(test_str)
        test_token_ids = [441, 1857, 4174, 11, 5242, 366]
        expected_str = self.hf_processor.tokenizer.decode(test_token_ids, skip_special_tokens=False)
        self.assertEqual(fn_decoder(np.asarray(test_token_ids)), expected_str)


if __name__ == "__main__":
    unittest.main()
