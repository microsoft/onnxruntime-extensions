// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// NeMo-compatible log-mel spectrogram extraction (Slaney scale, matching librosa/NeMo).

#include "nemo_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <dlib/matrix.h>
#include <math/dlib/stft_norm.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace nemo_mel {

// Apply pre-emphasis filter: y[n] = x[n] - preemph * x[n-1]
// For batch mode pass prev_sample = 0.0f; for streaming pass the last sample
// from the previous chunk. Returns audio[n-1] (the new prev_sample for streaming).
float ApplyPreemphasis(const float* audio, size_t n, float preemph,
                       float prev_sample, float* out) {
  if (n == 0) return prev_sample;
  out[0] = audio[0] - preemph * prev_sample;
  for (size_t i = 1; i < n; ++i) {
    out[i] = audio[i] - preemph * audio[i - 1];
  }
  return audio[n - 1];
}

// Slaney mel scale constants
static constexpr float kMinLogHz = 1000.0f;
static constexpr float kMinLogMel = 15.0f;               // 1000 / (200/3)
static constexpr float kLinScale = 200.0f / 3.0f;        // Hz per mel (linear region)
static constexpr float kLogStep = 0.06875177742094912f;   // log(6.4) / 27

float HzToMel(float hz) {
  if (hz < kMinLogHz) return hz / kLinScale;
  return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
}

float MelToHz(float mel) {
  if (mel < kMinLogMel) return mel * kLinScale;
  return kMinLogHz * std::exp((mel - kMinLogMel) * kLogStep);
}

std::vector<std::vector<float>> CreateMelFilterbank(int num_mels, int fft_size, int sample_rate) {
  int num_bins = fft_size / 2 + 1;
  float mel_low = HzToMel(0.0f);
  float mel_high = HzToMel(static_cast<float>(sample_rate) / 2.0f);

  // Compute mel center frequencies in Hz (num_mels + 2 points)
  std::vector<float> mel_f(num_mels + 2);
  for (int i = 0; i < num_mels + 2; ++i) {
    float m = mel_low + (mel_high - mel_low) * i / (num_mels + 1);
    mel_f[i] = MelToHz(m);
  }

  // Differences between consecutive mel center frequencies (Hz)
  std::vector<float> fdiff(num_mels + 1);
  for (int i = 0; i < num_mels + 1; ++i) {
    fdiff[i] = mel_f[i + 1] - mel_f[i];
  }

  // FFT bin center frequencies in Hz
  std::vector<float> fft_freqs(num_bins);
  for (int k = 0; k < num_bins; ++k) {
    fft_freqs[k] = static_cast<float>(k) * sample_rate / fft_size;
  }

  // Build triangular filterbank with Slaney normalization (matches librosa exactly)
  std::vector<std::vector<float>> filterbank(num_mels, std::vector<float>(num_bins, 0.0f));
  for (int m = 0; m < num_mels; ++m) {
    for (int k = 0; k < num_bins; ++k) {
      float lower = (fft_freqs[k] - mel_f[m]) / (fdiff[m] + 1e-10f);
      float upper = (mel_f[m + 2] - fft_freqs[k]) / (fdiff[m + 1] + 1e-10f);
      filterbank[m][k] = std::max(0.0f, std::min(lower, upper));
    }
    // Slaney area normalization: 2 / bandwidth
    float enorm = 2.0f / (mel_f[m + 2] - mel_f[m] + 1e-10f);
    for (int k = 0; k < num_bins; ++k) {
      filterbank[m][k] *= enorm;
    }
  }
  return filterbank;
}

void ComputeSTFTFrame(const float* frame, const float* window, int frame_len,
                      int fft_size, std::vector<float>& magnitudes) {
  int num_bins = fft_size / 2 + 1;
  magnitudes.resize(num_bins);

  // Apply window and zero-pad to fft_size for FFT
  dlib::matrix<float, 1, 0> windowed(1, fft_size);
  windowed = 0;
  for (int n = 0; n < frame_len; ++n) {
    windowed(0, n) = frame[n] * window[n];
  }

  // Fast FFT via dlib (returns num_bins complex values for real input)
  dlib::matrix<std::complex<float>> fft_result = dlib::fftr(windowed);

  // Calculate power spectrum
  for (int k = 0; k < num_bins; ++k) {
    float re = fft_result(0, k).real();
    float im = fft_result(0, k).imag();
    magnitudes[k] = re * re + im * im;
  }
}

std::vector<float> NemoComputeLogMelBatch(const float* audio, size_t num_samples,
                                          const NemoMelConfig& cfg, int& out_num_frames) {
  static auto mel_filters = CreateMelFilterbank(cfg.num_mels, cfg.fft_size, cfg.sample_rate);
  static auto window = hann_window(cfg.win_length);

  int n = static_cast<int>(num_samples);

  // Apply pre-emphasis: y[n] = x[n] - preemph * x[n-1]
  std::vector<float> preemphasized(n);
  ApplyPreemphasis(audio, n, cfg.preemph, 0.0f, preemphasized.data());

  // Center-pad both sides: fft_size/2 zeros on each side (matching torch.stft center=True)
  int pad = cfg.fft_size / 2;
  std::vector<float> padded(pad + n + pad, 0.0f);
  if (n > 0) {
    std::memcpy(padded.data() + pad, preemphasized.data(), n * sizeof(float));
  }

  if (static_cast<int>(padded.size()) < cfg.fft_size) {
    padded.resize(cfg.fft_size, 0.0f);
  }

  // Frame count using fft_size as frame size (matching torch.stft)
  int num_frames = static_cast<int>((padded.size() - cfg.fft_size) / cfg.hop_length) + 1;
  out_num_frames = num_frames;

  int win_offset = (cfg.fft_size - cfg.win_length) / 2;
  int num_bins = cfg.fft_size / 2 + 1;
  std::vector<float> magnitudes;
  std::vector<float> mel_spec(cfg.num_mels * num_frames);

  for (int t = 0; t < num_frames; ++t) {
    const float* frame = padded.data() + t * cfg.hop_length + win_offset;
    ComputeSTFTFrame(frame, window.data(), cfg.win_length, cfg.fft_size, magnitudes);

    for (int m = 0; m < cfg.num_mels; ++m) {
      float val = 0.0f;
      for (int k = 0; k < num_bins; ++k) {
        val += mel_filters[m][k] * magnitudes[k];
      }
      mel_spec[m * num_frames + t] = std::log(val + cfg.log_eps);
    }
  }

  return mel_spec;
}

NemoStreamingMelExtractor::NemoStreamingMelExtractor(const NemoMelConfig& cfg)
    : cfg_(cfg) {
  mel_filters_ = CreateMelFilterbank(cfg_.num_mels, cfg_.fft_size, cfg_.sample_rate);
  hann_window_ = hann_window_symmetric(cfg_.win_length);
  audio_overlap_.assign(cfg_.fft_size / 2, 0.0f);
  preemph_last_sample_ = 0.0f;
}

void NemoStreamingMelExtractor::Reset() {
  audio_overlap_.assign(cfg_.fft_size / 2, 0.0f);
  preemph_last_sample_ = 0.0f;
}

std::pair<std::vector<float>, int> NemoStreamingMelExtractor::Process(
    const float* audio, size_t num_samples) {
  // Apply pre-emphasis filter: y[n] = x[n] - preemph * x[n-1]
  std::vector<float> preemphasized(num_samples);
  preemph_last_sample_ = ApplyPreemphasis(audio, num_samples, cfg_.preemph,
                                          preemph_last_sample_, preemphasized.data());

  // Left-only center pad for streaming: prepend overlap from previous chunk.
  // For the first chunk this is zeros.
  int pad = cfg_.fft_size / 2;
  std::vector<float> padded(pad + num_samples);
  std::memcpy(padded.data(), audio_overlap_.data(), pad * sizeof(float));
  std::memcpy(padded.data() + pad, preemphasized.data(), num_samples * sizeof(float));

  // Update overlap buffer for next chunk
  if (num_samples >= static_cast<size_t>(pad)) {
    audio_overlap_.assign(preemphasized.data() + num_samples - pad,
                          preemphasized.data() + num_samples);
  } else {
    size_t keep = pad - num_samples;
    std::vector<float> new_overlap(pad, 0.0f);
    std::memcpy(new_overlap.data(), audio_overlap_.data() + num_samples, keep * sizeof(float));
    std::memcpy(new_overlap.data() + keep, preemphasized.data(), num_samples * sizeof(float));
    audio_overlap_ = std::move(new_overlap);
  }

  int win_offset = (cfg_.fft_size - cfg_.win_length) / 2;
  padded.resize(padded.size() + win_offset, 0.0f);

  if (static_cast<int>(padded.size()) < win_offset + cfg_.win_length) {
    padded.resize(win_offset + cfg_.win_length, 0.0f);
  }

  int num_frames = static_cast<int>((padded.size() - win_offset - cfg_.win_length) / cfg_.hop_length) + 1;

  int num_bins = cfg_.fft_size / 2 + 1;
  std::vector<float> mel_spec(cfg_.num_mels * num_frames);

  for (int t = 0; t < num_frames; ++t) {
    const float* frame = padded.data() + t * cfg_.hop_length + win_offset;

    // FFT with symmetric Hann window (win_length samples, zero-padded to fft_size)
    std::vector<float> magnitudes;
    ComputeSTFTFrame(frame, hann_window_.data(), cfg_.win_length, cfg_.fft_size, magnitudes);

    // Apply mel filterbank + log
    for (int m = 0; m < cfg_.num_mels; ++m) {
      float val = 0.0f;
      for (int k = 0; k < num_bins; ++k) {
        val += mel_filters_[m][k] * magnitudes[k];
      }
      mel_spec[m * num_frames + t] = std::log(val + cfg_.log_eps);
    }
  }

  return {mel_spec, num_frames};
}

}  // namespace nemo_mel
