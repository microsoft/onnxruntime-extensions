import numpy as np
from onnxruntime_extensions import OrtPyFunction, StftNorm

# 1️⃣ Generate test signal (440 Hz sine wave)
sr = 16000
duration = 1.0
freq = 440.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
x = np.sin(2 * np.pi * freq * t).astype(np.float32)
print(f"Generated sine wave: {len(x)} samples @ {sr} Hz")

# 2️⃣ STFT parameters
frame_ms = 25
hop_ms = 10
n_fft = int(sr * frame_ms / 1000)
hop_length = int(sr * hop_ms / 1000)
window = np.hanning(n_fft).astype(np.float32)

# 3️⃣ Run STFTNorm custom op
stft_fn = OrtPyFunction.from_customop(StftNorm)
x_in = x[None, :]  # shape [1, num_samples]

spec = stft_fn(
    x_in,
    int(n_fft),
    int(hop_length),
    window,
    int(n_fft)
)

# 4️⃣ Inspect results
spec = np.squeeze(spec, axis=0)  # remove batch dim
print(f"Spectrogram shape: {spec.shape}")  # (freq_bins, frames)
print(f"Spectrogram dtype: {spec.dtype}")
print(f"Sample frame[0:5]:\n{spec[:5, 0]}")  # first 5 freq bins of first frame

# 5️⃣ Convert to dB just for inspection
spec_db = 20 * np.log10(spec + 1e-10)
print(f"dB range: min={spec_db.min():.2f}, max={spec_db.max():.2f}")
