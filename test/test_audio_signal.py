# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import wave
import unittest
import numpy as np
from onnxruntime_extensions import PyOrtFunction, util, make_onnx_model

import onnx
from onnx import onnx_pb as onnx_proto


_is_torch_available = False
try:
    import torch
    _is_torch_available = True
except ImportError:
    pass

_is_librosa_avaliable = False
try:
    import librosa
    _is_librosa_avaliable = True
except ImportError:
    pass


def _create_test_model(**kwargs):
    node = onnx.helper.make_node(
        "STFT",
        inputs=["signal", "nfft", "hop_length", "window", "frame_length"],
        outputs=["output"],
        domain='ai.onnx.contrib')

    input1 = onnx.helper.make_tensor_value_info(
        'signal', onnx_proto.TensorProto.FLOAT, [1, None])
    input2 = onnx.helper.make_tensor_value_info(
        'nfft', onnx_proto.TensorProto.INT64, [])
    input3 = onnx.helper.make_tensor_value_info(
        'hop_length', onnx_proto.TensorProto.INT64, [])
    input4 = onnx.helper.make_tensor_value_info(
        'window', onnx_proto.TensorProto.FLOAT, [None])
    input5 = onnx.helper.make_tensor_value_info(
        'frame_length', onnx_proto.TensorProto.INT64, [])
    output1 = onnx.helper.make_tensor_value_info(
        'output', onnx_proto.TensorProto.FLOAT, [1, None, None, 2])

    graph = onnx.helper.make_graph([node], 'test0', [input1, input2, input3, input4, input5], [output1])
    model = make_onnx_model(graph)
    return model


class TestAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wavefile = wave.open(util.get_test_data_file('data/1272-141231-0002.wav'), 'r')
        samples = wavefile.getnframes()
        audio = wavefile.readframes(samples)
        audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
        max_int16 = 2**15
        cls.test_pcm = audio_as_np_float32 / max_int16

    @staticmethod
    def stft(waveform, n_fft, hop_length, window):
        """
        Compute the short-time Fourier transform (STFT) of a signal.
        Should be equivalent to torch.stft
        """
        frames = []
        for i in range(0, waveform.shape[0] + 1, hop_length):
            half_window = (n_fft - 1) // 2 + 1
            start = i - half_window if i > half_window else 0
            end = i + half_window if i < waveform.shape[0] - half_window else waveform.shape[0]

            frame = waveform[start:end]

            if start == 0:
                padd_width = (-i + half_window, 0)
                frame = np.pad(frame, pad_width=padd_width, mode="reflect")

            elif end == waveform.shape[0]:
                padd_width = (0, (i - waveform.shape[0] + half_window))
                frame = np.pad(frame, pad_width=padd_width, mode="reflect")
            frames.append(frame)

        fft_size = n_fft
        frame_size = fft_size
        num_fft_bins = (fft_size >> 1) + 1

        data = np.empty((len(frames), num_fft_bins), dtype=np.complex64)
        fft_signal = np.zeros(fft_size)

        for f, frame in enumerate(frames):
            np.multiply(frame, window, out=fft_signal[:frame_size])
            data[f] = np.fft.fft(fft_signal, axis=0)[:num_fft_bins]
        return np.absolute(data.T) ** 2

    def test_onnx_stft(self):
        audio_pcm = self.test_pcm
        expected = self.stft(audio_pcm, 400, 160, np.hanning(400).astype(np.float32))

        ortx_stft = PyOrtFunction.from_model(_create_test_model(), cpu_only=True)
        actual = ortx_stft(np.expand_dims(audio_pcm, axis=0), 400, 160, np.hanning(400).astype(np.float32), 400)
        actual = actual[0]
        actual = actual[:, :, 0] ** 2 + actual[:, :, 1] ** 2
        np.testing.assert_allclose(expected[:, 1:], actual[:, 1:], rtol=1e-3, atol=1e-3)

    def test_stft_norm_np(self):
        audio_pcm = self.test_pcm
        expected = self.stft(audio_pcm, 400, 160, np.hanning(400).astype(np.float32))

        ortx_stft = PyOrtFunction.from_customop("StftNorm", cpu_only=True)
        actual = ortx_stft(np.expand_dims(audio_pcm, axis=0), 400, 160, np.hanning(400).astype(np.float32), 400)
        actual = actual[0]
        np.testing.assert_allclose(expected[:, 1:], actual[:, 1:], rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not _is_torch_available, "PyTorch is not available")
    def test_stft_norm_torch(self):
        audio_pcm = self.test_pcm
        wlen = 400
        # intesting bug in torch.stft, if there is 2-D input with batch size 1, it will generate a different
        # result with some spark points in the spectrogram.
        expected = torch.stft(torch.from_numpy(audio_pcm),
                              400, 160, wlen, torch.from_numpy(np.hanning(wlen).astype(np.float32)),
                              center=True,
                              return_complex=True).abs().pow(2).numpy()
        audio_pcm = np.expand_dims(self.test_pcm, axis=0)
        ortx_stft = PyOrtFunction.from_customop("StftNorm")
        actual = ortx_stft(audio_pcm, 400, 160, np.hanning(wlen).astype(np.float32), 400)
        actual = actual[0]
        np.testing.assert_allclose(expected[:, 1:], actual[:, 1:], rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not _is_librosa_avaliable, "librosa is not available")
    def test_mel_filter_bank(self):
        expected = librosa.filters.mel(n_fft=400, n_mels=80, sr=16000)
        actual = util.mel_filterbank(400, 80, 16000)
        np.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
