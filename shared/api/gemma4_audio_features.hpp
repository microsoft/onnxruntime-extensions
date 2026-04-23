// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>

#include "ext_status.h"
#include "op_def_struct.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ort_extensions {

// Gemma 4 audio feature extraction: USM-style log-mel spectrogram that matches
// the HuggingFace Gemma4AudioFeatureExtractor exactly.
//
// Pipeline:  AudioDecoder  ->  Gemma4LogMel
//
// Inputs:   float (1, num_samples)  — mono PCM at `sampling_rate` Hz
// Outputs:  float (num_frames, feature_size) — log-mel features
//           bool  (num_frames,)              — frame-level attention mask
class Gemma4LogMel {
 public:
  Gemma4LogMel() = default;

  // ---------- HTK mel-frequency conversions --------------------------------
  static double HzToMel(double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }
  static double MelToHz(double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); }

  // Build an HTK-scale mel filterbank  (num_freq_bins x num_mel_filters).
  static std::vector<float> MelFilterBank(int num_freq_bins, int num_mel_filters,
                                          int sampling_rate,
                                          double min_freq, double max_freq) {
    const double mel_min = HzToMel(min_freq);
    const double mel_max = HzToMel(max_freq);
    const int n_points = num_mel_filters + 2;
    std::vector<double> mel_points(n_points);
    for (int i = 0; i < n_points; ++i) {
      mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_points - 1);
    }

    // Convert mel points back to Hz, then to FFT bin indices.
    std::vector<double> freq_points(n_points);
    for (int i = 0; i < n_points; ++i) {
      freq_points[i] = MelToHz(mel_points[i]);
    }

    // Filterbank: shape (num_freq_bins, num_mel_filters), stored row-major.
    std::vector<float> fb(static_cast<size_t>(num_freq_bins) * num_mel_filters, 0.0f);

    auto hz_to_bin = [&](double hz) -> double {
      return hz * (num_freq_bins - 1) * 2.0 / sampling_rate;
    };

    for (int m = 0; m < num_mel_filters; ++m) {
      const double left = hz_to_bin(freq_points[m]);
      const double center = hz_to_bin(freq_points[m + 1]);
      const double right = hz_to_bin(freq_points[m + 2]);

      for (int k = 0; k < num_freq_bins; ++k) {
        double kd = static_cast<double>(k);
        float w = 0.0f;
        if (kd > left && kd <= center && center > left) {
          w = static_cast<float>((kd - left) / (center - left));
        } else if (kd > center && kd < right && right > center) {
          w = static_cast<float>((right - kd) / (right - center));
        }
        fb[static_cast<size_t>(k) * num_mel_filters + m] = w;
      }
    }
    return fb;
  }

  // ---------- Periodic Hann window -----------------------------------------
  static std::vector<float> PeriodicHannWindow(int N) {
    // w[n] = 0.5 - 0.5 * cos(2*pi*n / N)   for n = 0 .. N-1
    std::vector<float> w(N);
    for (int n = 0; n < N; ++n) {
      double s = std::sin(M_PI * n / N);
      w[n] = static_cast<float>(s * s);
    }
    return w;
  }

  // ---------- Real FFT (magnitude only) ------------------------------------
  // Cooley-Tukey radix-2 DIT FFT, returns only non-negative frequency magnitudes.
  static void RealFFTMagnitude(const float* data, int n_fft, std::vector<float>& mag) {
    const int n_out = n_fft / 2 + 1;
    mag.resize(n_out);

    // Build complex input (zero-padded if data is shorter than n_fft).
    std::vector<std::complex<double>> x(n_fft, {0.0, 0.0});
    // Caller must ensure the data pointer has at least n_fft valid elements
    // (it may have been zero-padded externally).
    for (int i = 0; i < n_fft; ++i) {
      x[i] = {static_cast<double>(data[i]), 0.0};
    }

    // Bit-reversal permutation.
    int log2n = 0;
    for (int tmp = n_fft; tmp > 1; tmp >>= 1) ++log2n;
    for (int i = 1, j = 0; i < n_fft; ++i) {
      int bit = n_fft >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) std::swap(x[i], x[j]);
    }

    // Butterfly stages.
    for (int len = 2; len <= n_fft; len <<= 1) {
      double ang = -2.0 * M_PI / len;
      std::complex<double> wlen(std::cos(ang), std::sin(ang));
      for (int i = 0; i < n_fft; i += len) {
        std::complex<double> w(1.0, 0.0);
        for (int j = 0; j < len / 2; ++j) {
          auto u = x[i + j];
          auto v = x[i + j + len / 2] * w;
          x[i + j] = u + v;
          x[i + j + len / 2] = u - v;
          w *= wlen;
        }
      }
    }

    for (int i = 0; i < n_out; ++i) {
      mag[i] = static_cast<float>(std::abs(x[i]));
    }
  }

  // ---------- Main Compute -------------------------------------------------
  OrtxStatus Compute(const ortc::Tensor<float>& pcm_input,
                     ortc::Tensor<float>& logmel_out,
                     ortc::Tensor<bool>& mask_out) {
    const auto& pcm_shape = pcm_input.Shape();
    if (pcm_shape.size() != 2 || pcm_shape[0] != 1) {
      return {kOrtxErrorInvalidArgument,
              "[Gemma4LogMel]: expected (1, num_samples) float input"};
    }

    const int64_t num_samples = pcm_shape[1];
    const float* pcm = pcm_input.Data();

    // --- semicausal padding ------------------------------------------------
    const int64_t pad_left = frame_length_ / 2;
    const int64_t padded_len = num_samples + pad_left;
    std::vector<float> padded(static_cast<size_t>(padded_len), 0.0f);
    std::copy(pcm, pcm + num_samples, padded.begin() + pad_left);

    // Build a per-sample attention mask (1 = real audio, 0 = padding).
    std::vector<uint8_t> sample_mask(static_cast<size_t>(padded_len), 0);
    std::fill(sample_mask.begin() + pad_left,
              sample_mask.begin() + pad_left + num_samples, 1);

    // --- unfold into overlapping frames ------------------------------------
    const int64_t frame_size_unfold = frame_length_ + 1;  // 321 @ 16 kHz
    if (padded_len < frame_size_unfold) {
      // Audio too short to form even one frame — return empty tensors.
      logmel_out.Allocate({0, feature_size_});
      mask_out.Allocate({0});
      return {};
    }
    const int64_t num_frames = (padded_len - frame_size_unfold) / hop_length_ + 1;

    // --- per-frame processing ----------------------------------------------
    const int n_freq = fft_length_ / 2 + 1;
    std::vector<float> logmel_data(static_cast<size_t>(num_frames * feature_size_));
    std::vector<bool> frame_mask(static_cast<size_t>(num_frames));

    std::vector<float> frame_buf(fft_length_, 0.0f);
    std::vector<float> mag;
    std::vector<float> processed(static_cast<size_t>(frame_length_));

    for (int64_t fi = 0; fi < num_frames; ++fi) {
      const int64_t offset = fi * hop_length_;
      const float* frame_start = padded.data() + offset;

      // --- pre-emphasis (HTK flavour) or simple truncation -----------------
      if (preemphasis_ > 0.0f) {
        if (preemphasis_htk_flavor_) {
          processed[0] = frame_start[0] * (1.0f - preemphasis_);
          for (int64_t i = 1; i < frame_length_; ++i) {
            processed[i] = frame_start[i] - preemphasis_ * frame_start[i - 1];
          }
        } else {
          for (int64_t i = 0; i < frame_length_; ++i) {
            processed[i] = frame_start[i + 1] - preemphasis_ * frame_start[i];
          }
        }
      } else {
        // No preemphasis — drop the last sample of the unfolded frame.
        std::copy(frame_start, frame_start + frame_length_, processed.begin());
      }

      // --- window ----------------------------------------------------------
      for (int64_t i = 0; i < frame_length_; ++i) {
        processed[i] *= window_[i];
      }

      // --- zero-pad to fft_length and compute magnitude FFT ----------------
      std::fill(frame_buf.begin(), frame_buf.end(), 0.0f);
      std::copy(processed.begin(), processed.end(), frame_buf.begin());
      RealFFTMagnitude(frame_buf.data(), fft_length_, mag);

      // --- mel filterbank + log --------------------------------------------
      float* mel_row = logmel_data.data() + fi * feature_size_;
      for (int64_t m = 0; m < feature_size_; ++m) {
        double sum = 0.0;
        for (int k = 0; k < n_freq; ++k) {
          sum += static_cast<double>(mag[k]) *
                 mel_filters_[static_cast<size_t>(k) * feature_size_ + m];
        }
        mel_row[m] = std::log(static_cast<float>(sum) + mel_floor_);
      }

      // --- frame-level attention mask --------------------------------------
      // A frame is valid when the last sample of its window is real audio.
      const int64_t frame_end_idx = offset + frame_size_unfold - 1;
      frame_mask[fi] = (frame_end_idx < padded_len) && sample_mask[frame_end_idx];
    }

    // --- zero out padding frames -------------------------------------------
    for (int64_t fi = 0; fi < num_frames; ++fi) {
      if (!frame_mask[fi]) {
        float* mel_row = logmel_data.data() + fi * feature_size_;
        std::fill(mel_row, mel_row + feature_size_, 0.0f);
      }
    }

    // --- per-bin normalization (optional) -----------------------------------
    if (!per_bin_mean_.empty()) {
      for (int64_t fi = 0; fi < num_frames; ++fi) {
        float* mel_row = logmel_data.data() + fi * feature_size_;
        for (int64_t m = 0; m < feature_size_; ++m) {
          mel_row[m] -= per_bin_mean_[m];
        }
      }
    }
    if (!per_bin_stddev_.empty()) {
      for (int64_t fi = 0; fi < num_frames; ++fi) {
        float* mel_row = logmel_data.data() + fi * feature_size_;
        for (int64_t m = 0; m < feature_size_; ++m) {
          if (per_bin_stddev_[m] != 0.0f) {
            mel_row[m] /= per_bin_stddev_[m];
          }
        }
      }
    }

    // --- write outputs -----------------------------------------------------
    float* out_mel = logmel_out.Allocate({num_frames, feature_size_});
    std::copy(logmel_data.begin(), logmel_data.end(), out_mel);

    bool* out_mask = mask_out.Allocate({num_frames});
    for (int64_t i = 0; i < num_frames; ++i) {
      out_mask[i] = frame_mask[i];
    }

    return {};
  }

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "feature_size") {
        feature_size_ = std::get<int64_t>(value);
      } else if (key == "sampling_rate") {
        sampling_rate_ = std::get<int64_t>(value);
      } else if (key == "frame_length_ms") {
        frame_length_ms_ = std::get<double>(value);
      } else if (key == "hop_length_ms") {
        hop_length_ms_ = std::get<double>(value);
      } else if (key == "min_frequency") {
        min_frequency_ = std::get<double>(value);
      } else if (key == "max_frequency") {
        max_frequency_ = std::get<double>(value);
      } else if (key == "preemphasis") {
        preemphasis_ = static_cast<float>(std::get<double>(value));
      } else if (key == "preemphasis_htk_flavor") {
        preemphasis_htk_flavor_ = std::get<int64_t>(value) != 0;
      } else if (key == "fft_overdrive") {
        fft_overdrive_ = std::get<int64_t>(value) != 0;
      } else if (key == "mel_floor") {
        mel_floor_ = static_cast<float>(std::get<double>(value));
      } else if (key == "per_bin_mean") {
        auto& v = std::get<std::vector<double>>(value);
        per_bin_mean_.assign(v.begin(), v.end());
      } else if (key == "per_bin_stddev") {
        auto& v = std::get<std::vector<double>>(value);
        per_bin_stddev_.assign(v.begin(), v.end());
      } else {
        return {kOrtxErrorInvalidArgument,
                "[Gemma4LogMel]: unknown attribute '" + key + "'"};
      }
    }

    // Derive sample-domain lengths from millisecond config.
    frame_length_ = static_cast<int64_t>(
        std::round(sampling_rate_ * frame_length_ms_ / 1000.0));
    hop_length_ = static_cast<int64_t>(
        std::round(sampling_rate_ * hop_length_ms_ / 1000.0));

    // Pre-compute window, FFT length, and mel filterbank so Compute() is
    // allocation-free for these (avoids lazy-init races if reused concurrently).
    window_ = PeriodicHannWindow(static_cast<int>(frame_length_));
    int fft_len = 1;
    while (fft_len < frame_length_) fft_len <<= 1;
    if (fft_overdrive_) fft_len <<= 1;
    fft_length_ = fft_len;

    mel_filters_ = MelFilterBank(
        fft_len / 2 + 1, static_cast<int>(feature_size_),
        static_cast<int>(sampling_rate_), min_frequency_, max_frequency_);

    return {};
  }

 private:
  // Configuration (matching Gemma4AudioFeatureExtractor defaults).
  int64_t feature_size_ = 128;
  int64_t sampling_rate_ = 16000;
  double frame_length_ms_ = 20.0;
  double hop_length_ms_ = 10.0;
  double min_frequency_ = 0.0;
  double max_frequency_ = 8000.0;
  float preemphasis_ = 0.0f;
  bool preemphasis_htk_flavor_ = true;
  bool fft_overdrive_ = false;
  float mel_floor_ = 0.001f;
  std::vector<float> per_bin_mean_;
  std::vector<float> per_bin_stddev_;

  // Derived / cached state (computed in Init()).
  int64_t frame_length_ = 320;   // samples
  int64_t hop_length_ = 160;     // samples
  int fft_length_ = 512;
  std::vector<float> window_;
  std::vector<float> mel_filters_;  // (n_freq x feature_size), row-major
};

}  // namespace ort_extensions
