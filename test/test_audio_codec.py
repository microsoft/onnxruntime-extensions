import unittest
import numpy as np

from onnxruntime_extensions import PyOrtFunction, util


class TestBpeTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_mp3_file = util.get_test_data_file('data', 'speech16k.mp3')
        return super().setUpClass()

    def test_mp3_decoder(self):
        decoder = PyOrtFunction.from_customop('AudioDecoder')
        blob = []
        with open(self.test_mp3_file, 'rb') as _f:
            blob = bytearray(_f.read())

        pcm_tensor = decoder(np.expand_dims(np.asarray(blob), axis=(0,)))
        self.assertEqual(pcm_tensor.shape[0], 1)
        self.assertTrue(pcm_tensor.shape[1] > len(blob))


if __name__ == "__main__":
    unittest.main()
