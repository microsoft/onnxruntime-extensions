import numpy as np
import torch
import librosa
from onnxruntime_extensions import OrtPyFunction, StftNorm

# ============================
# 1️⃣ Load real audio (60s–120s slice)
# ============================
wav_path = "test/kirk.mp3"  # 👈 your real file path
audio, sr = librosa.load(wav_path, sr=16000, mono=True)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)  # stereo → mono

# Trim to 60–120 seconds
start_s = 60
end_s = 120
start_sample = int(start_s * sr)
end_sample = int(end_s * sr)
audio = audio[start_sample:end_sample]

audio = audio.astype(np.float32)
print(f"Loaded audio segment: {wav_path}, {len(audio)} samples ({start_s}s–{end_s}s) @ {sr} Hz")

# ============================
# 2️⃣ STFT parameters
# ============================
frame_ms = 25
hop_ms = 10
n_fft = int(sr * frame_ms / 1000)
hop_length = int(sr * hop_ms / 1000)
window_torch = torch.hann_window(n_fft)
window_np = np.hanning(n_fft).astype(np.float32)

# ============================
# 3️⃣ Torch STFT
# ============================
x_t = torch.tensor(audio)
spec_complex_torch = torch.stft(
    x_t,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=n_fft,
    window=window_torch,
    center=False,
    return_complex=True
)
mag_torch = spec_complex_torch.abs().numpy()
db_torch = 40 * np.log10(mag_torch + 1e-10)

# ============================
# 4️⃣ C++ StftNorm
# ============================
stft_fn = OrtPyFunction.from_customop(StftNorm)
audio_in = audio[None, :]  # shape [1, num_samples]

spec_cpp = stft_fn(
    audio_in,
    int(n_fft),
    int(hop_length),
    window_np,
    int(n_fft)
)
spec_cpp = np.squeeze(spec_cpp, axis=0)
db_cpp = 20 * np.log10(spec_cpp + 1e-10)  # spec_cpp is already power

print(f"Torch shape={db_torch.shape}, C++ shape={db_cpp.shape}")

# ============================
# 5️⃣ Segment detection helper
# ============================
def detect_segments(energy_db, hop_length, sr, thresh=-40):
    frame_energy = energy_db.mean(axis=0)
    active = frame_energy > thresh
    segments = []
    start = None
    for i, val in enumerate(active):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i
            segments.append((start, end))
            start = None
    if start is not None:
        segments.append((start, len(active)))

    return [(s * hop_length / sr, e * hop_length / sr) for s, e in segments]

# ============================
# 6️⃣ Compare segments
# ============================
segments_torch = detect_segments(db_torch, hop_length, sr)
segments_cpp = detect_segments(db_cpp, hop_length, sr)

print("\nTorch detected segments (60–120s):")
#for s, e in segments_torch:
    #print(f"  {s + start_s:.2f}s – {e + start_s:.2f}s")

print("\nC++ StftNorm detected segments (60–120s):")
#for s, e in segments_cpp:
    #print(f"  {s + start_s:.2f}s – {e + start_s:.2f}s")
