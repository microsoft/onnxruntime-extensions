import wave
import unittest
import numpy as np
from onnxruntime_extensions import PyOrtFunction, util


_is_torch_available = False
try:
    import torch
    _is_torch_available = True
except ImportError:
    pass


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

    def test_stft_norm_np(self):
        audio_pcm = self.test_pcm
        expected = self.stft(audio_pcm, 400, 160, np.hanning(400).astype(np.float32))

        ortx_stft = PyOrtFunction.from_customop("StftNorm")
        actual = ortx_stft(audio_pcm, 400, 160, 400, np.hanning(400).astype(np.float32))
        np.testing.assert_allclose(expected[:, 1:], actual[:, 1:], rtol=1e-3, atol=1e-3)

    @unittest.skipIf(not _is_torch_available, "PyTorch is not available")
    def test_stft_norm_torch(self):
        audio_pcm = self.test_pcm
        wlen = 400
        expected = torch.stft(torch.from_numpy(audio_pcm),
                              400, 160, wlen, torch.from_numpy(np.hanning(wlen).astype(np.float32)),
                              center=True,
                              return_complex=True).abs().pow(2).numpy()
        ortx_stft = PyOrtFunction.from_customop("StftNorm")
        actual = ortx_stft(audio_pcm, 400, 160, wlen, np.hanning(wlen).astype(np.float32))
        np.testing.assert_allclose(expected[:, 1:], actual[:, 1:], rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()