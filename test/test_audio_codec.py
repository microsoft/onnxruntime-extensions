# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
import numpy as np

from onnx import checker, helper, onnx_pb as onnx_proto
from onnxruntime_extensions import PyOrtFunction, util


class TestAudioCodec(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_mp3_file = util.get_test_data_file('data', '1272-141231-0002.mp3')
        cls.test_wav_file = util.get_test_data_file('data', '1272-141231-0002.wav')
        cls.test_flac_file = util.get_test_data_file('data', '1272-141231-0002.flac')
        cls.raw_data = np.load(util.get_test_data_file('data', '1272-141231-0002.npy'))
        cls.decoder = PyOrtFunction.from_customop('AudioDecoder')
        return super().setUpClass()

    def test_wav_decoder(self):
        blob = bytearray(util.read_file(self.test_wav_file, mode='rb'))
        pcm_tensor = self.decoder(np.expand_dims(np.asarray(blob), axis=(0,)))
        np.testing.assert_allclose(pcm_tensor, self.raw_data, rtol=1e-05, atol=1e-08)

    def test_wav_format_decoder(self):
        blob = bytearray(util.read_file(self.test_wav_file, mode='rb'))
        fmt_onnx_model = self.decoder.onnx_model
        fmt_onnx_model.graph.input.extend([
            helper.make_tensor_value_info('format', onnx_proto.TensorProto.STRING, [])])
        fmt_onnx_model.graph.node[0].input.extend(['format'])
        checker.check_model(fmt_onnx_model)
        new_decoder = PyOrtFunction.from_model(fmt_onnx_model)
        pcm_tensor = new_decoder(np.expand_dims(np.asarray(blob), axis=(0,)), ["wav"])
        np.testing.assert_allclose(pcm_tensor, self.raw_data, rtol=1e-05, atol=1e-08)

    def test_flac_decoder(self):
        blob = bytearray(util.read_file(self.test_flac_file, mode='rb'))
        pcm_tensor = self.decoder(np.expand_dims(np.asarray(blob), axis=(0,)))
        np.testing.assert_allclose(pcm_tensor, self.raw_data, atol=1e-03)

    def test_mp3_decoder(self):
        blob = bytearray(util.read_file(self.test_mp3_file, mode='rb'))
        pcm_tensor = self.decoder(np.expand_dims(np.asarray(blob), axis=(0,)))
        self.assertTrue(pcm_tensor.shape[1] > len(blob))
        # lossy compression, so we can only check the range
        np.testing.assert_allclose(
            np.asarray([np.max(pcm_tensor), np.average(pcm_tensor), np.min(pcm_tensor)]),
            np.asarray([np.max(self.raw_data), np.average(self.raw_data), np.min(self.raw_data)]), atol=1e-01)

    def test_decoder_resampling(self):
        test_file = util.get_test_data_file('data', 'jfk.flac')
        blob = bytearray(util.read_file(test_file, mode='rb'))
        decoder = PyOrtFunction.from_customop('AudioDecoder', cpu_only=True, downsampling_rate=16000, stereo_to_mono=1)
        pcm_tensor = decoder(np.expand_dims(np.asarray(blob), axis=(0,)))
        self.assertEqual(pcm_tensor.shape, (1, 176000))


if __name__ == "__main__":
    unittest.main()
