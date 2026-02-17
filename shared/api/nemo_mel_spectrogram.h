// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// NeMo-compatible log-mel spectrogram extraction (Slaney scale, matching librosa/NeMo).

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace nemo_mel {

struct NemoMelConfig {
  int num_mels;
  int fft_size;
  int hop_length;
  int win_length;
  int sample_rate;
  float preemph;
  float log_eps;
};

// Mel scale conversions (Slaney)

float HzToMel(float hz);
float MelToHz(float mel);

/// Build a triangular mel filterbank with Slaney normalization (matches librosa).
/// Returns shape [num_mels][num_bins] where num_bins = fft_size/2 + 1.
std::vector<std::vector<float>> CreateMelFilterbank(int num_mels, int fft_size, int sample_rate);

/// Apply pre-emphasis filter: y[n] = x[n] - preemph * x[n-1].
/// For batch mode pass prev_sample = 0.0f; for streaming pass the last sample
/// from the previous chunk.
/// Returns audio[n-1] (the new prev_sample for the next streaming chunk).
float ApplyPreemphasis(const float* audio, size_t n, float preemph,
                       float prev_sample, float* out);

/// Compute |DFT|^2 (power spectrum) for a single windowed frame.
/// frame: pointer to fft_size samples (or win_length samples with window applied).
/// window: pointer to window coefficients (same length as frame_len).
/// frame_len: number of samples to read from frame and window.
/// fft_size: DFT size (output has fft_size/2 + 1 bins).
/// magnitudes: output power spectrum (resized to fft_size/2 + 1).
void ComputeSTFTFrame(const float* frame, const float* window, int frame_len,
                      int fft_size, std::vector<float>& magnitudes);


// BATCH LOG-MEL EXTRACTION
/// Compute NeMo-compatible log-mel spectrogram for a complete audio buffer.
/// Applies pre-emphasis, center-pads both sides (fft_size/2 zeros), computes STFT
/// with a periodic Hann window, applies mel filterbank, and takes log(mel + eps).
///
/// Output layout: row-major [num_mels, num_frames].
/// out_num_frames is set to the number of time frames produced.
std::vector<float> NemoComputeLogMelBatch(const float* audio, size_t num_samples,
                                          const NemoMelConfig& cfg, int& out_num_frames);

// STREAMING LOG-MEL EXTRACTION
/// Stateful streaming NeMo-compatible mel extractor that maintains overlap and
/// pre-emphasis state across successive audio chunks.
///
/// Usage:
///   nemo_mel::NemoStreamingMelExtractor extractor(cfg);
///   auto [mel, frames] = extractor.Process(chunk1, n1);
///   auto [mel2, frames2] = extractor.Process(chunk2, n2);
///   extractor.Reset();  // new utterance
///
class NemoStreamingMelExtractor {
 public:
  explicit NemoStreamingMelExtractor(const NemoMelConfig& cfg);

  /// Process one chunk of raw PCM audio (mono, float32).
  /// Returns (mel_data, num_frames) where mel_data is row-major [num_mels, num_frames].
  std::pair<std::vector<float>, int> Process(const float* audio, size_t num_samples);

  /// Reset all streaming state for a new utterance.
  void Reset();

  const NemoMelConfig& config() const { return cfg_; }

 private:
  NemoMelConfig cfg_;
  std::vector<std::vector<float>> mel_filters_;
  std::vector<float> hann_window_;  // symmetric, length = win_length

  // Streaming state
  std::vector<float> audio_overlap_;   // last fft_size/2 pre-emphasized samples
  float preemph_last_sample_{0.0f};
};

}  // namespace nemo_mel
