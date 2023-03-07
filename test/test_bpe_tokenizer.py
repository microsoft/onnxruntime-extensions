import unittest
import numpy as np

from transformers import AutoProcessor
from onnxruntime_extensions import PyOrtFunction
from onnxruntime_extensions.cvt import HFTokenizerConverter


class TestBpeTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.hf_processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        cls.processor = HFTokenizerConverter(cls.hf_processor)
        return super().setUpClass()
    
    @unittest.skip("Not implemented")
    def test_bpe_tokenizer(self):
        fn_tokenizer = PyOrtFunction.from_customop("GPT2Tokenizer",
                                        cvt=(type(self).processor).bbpe_tokenizer)

        # fmt: off
        test_str = [
            441, 1857, 4174, 11, 5242, 366,
            257, 1333, 295, 493, 2794, 2287, 293, 12018, 14880, 11,
            293, 25730, 311, 454, 34152, 4496, 904, 50724
        ]
        # fmt: on

        self.assertTrue(fn_tokenizer(test_str),
                        " Lennils, pictures are a sort of upguards and atom paintings, and Mason's exquisite idles")

    def test_en_decoder(self):
        fn_decoder = PyOrtFunction.from_customop("BpeDecoder",
                                        cvt=type(self).processor.bpe_decoder,
                                        skip_special_tokens=True)

        test_token_ids = np.asarray([51492, 406, 3163, 1953, 466, 13, 51612, 51612])
        self.assertEqual(fn_decoder(test_token_ids), " not worth thinking about.")


if __name__ == "__main__":
    unittest.main()
