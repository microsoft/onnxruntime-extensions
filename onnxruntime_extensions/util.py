# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pathlib
import inspect

import numpy as np


# some util function for testing and tools
def get_test_data_file(*sub_dirs):
    case_file = inspect.currentframe().f_back.f_code.co_filename
    test_dir = pathlib.Path(case_file).parent
    return str(test_dir.joinpath(*sub_dirs).resolve())


def read_file(path, mode='r'):
    with open(str(path), mode) as file_content:
        return file_content.read()


def mel_filterbank(
        n_fft: int, n_mels: int = 80, sr=16000, min_mel=0, max_mel=45.245640471924965, dtype=np.float32):
    """
    Compute a Mel-filterbank. The filters are stored in the rows, the columns
    and it is Slaney normalized mel-scale filterbank.
    """
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=dtype)

    # the centers of the frequency bins for the DFT
    freq_bins = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    mel = np.linspace(min_mel, max_mel, n_mels + 2)
    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mel

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    log_t = mel >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mel[log_t] - min_log_mel))
    mel_bins = freqs

    mel_spacing = np.diff(mel_bins)

    ramps = mel_bins.reshape(-1, 1) - freq_bins.reshape(1, -1)
    for i in range(n_mels):
        left = -ramps[i] / mel_spacing[i]
        right = ramps[i + 2] / mel_spacing[i + 1]

        # intersect them with each other and zero
        fbank[i] = np.maximum(0, np.minimum(left, right))

    energy_norm = 2.0 / (mel_bins[2 : n_mels + 2] - mel_bins[:n_mels])
    fbank *= energy_norm[:, np.newaxis]
    return fbank
