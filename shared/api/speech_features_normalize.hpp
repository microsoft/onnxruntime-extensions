// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "nemo_mel_spectrogram.h"

#include <cmath>
#include <cstring>
#include <cstdint>

namespace ort_extensions {

// Per-feature (per-mel-bin) normalization: for each feature row,
// compute mean and std across time, then normalize.
// Input:  [1, num_features, num_frames]  (feature_first) or [1, num_frames, num_features]
// Output: same shape, normalized.
class PerFeatureNormalize {
 public:
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "eps") {
        eps_ = static_cast<float>(std::get<double>(value));
      } else if (key == "feature_first") {
        feature_first_ = std::get<int64_t>(value);
      } else if (key != "_comment") {
        return {kOrtxErrorInvalidArgument, "[PerFeatureNormalize]: Invalid key in the JSON configuration."};
      }
    }
    return {};
  }

  OrtxStatus Compute(const ortc::Tensor<float>& input, ortc::Tensor<float>& output) {
    const auto& shape = input.Shape();
    int64_t num_features, num_frames;

    if (shape.size() == 2) {
      // 2D: [features, frames] or [frames, features]
      num_features = feature_first_ ? shape[0] : shape[1];
      num_frames = feature_first_ ? shape[1] : shape[0];
    } else if (shape.size() == 3 && shape[0] == 1) {
      // 3D: [1, features, frames] or [1, frames, features]
      num_features = feature_first_ ? shape[1] : shape[2];
      num_frames = feature_first_ ? shape[2] : shape[1];
    } else {
      return {kOrtxErrorInvalidArgument, "[PerFeatureNormalize]: Expected input shape [features, frames] or [1, features, frames]."};
    }

    const float* in_data = input.Data();
    float* out_data = output.Allocate(shape);

    // Copy input to output first
    std::memcpy(out_data, in_data, num_features * num_frames * sizeof(float));

    // Need at least 2 frames for sample std (N-1 denominator)
    if (num_frames <= 1) {
      // Single frame or empty: output zeros (value - mean = 0 for constant)
      std::memset(out_data, 0, num_features * num_frames * sizeof(float));
      return {};
    }

    for (int64_t f = 0; f < num_features; ++f) {
      // Compute mean
      float sum = 0.0f;
      for (int64_t t = 0; t < num_frames; ++t) {
        int64_t idx = feature_first_ ? (f * num_frames + t) : (t * num_features + f);
        sum += out_data[idx];
      }
      float mean = sum / static_cast<float>(num_frames);

      // Compute std (sample std, divide by N-1)
      float var_sum = 0.0f;
      for (int64_t t = 0; t < num_frames; ++t) {
        int64_t idx = feature_first_ ? (f * num_frames + t) : (t * num_features + f);
        float d = out_data[idx] - mean;
        var_sum += d * d;
      }
      float std_val = std::sqrt(var_sum / static_cast<float>(num_frames - 1)) + eps_;

      // Normalize
      for (int64_t t = 0; t < num_frames; ++t) {
        int64_t idx = feature_first_ ? (f * num_frames + t) : (t * num_features + f);
        out_data[idx] = (out_data[idx] - mean) / std_val;
      }
    }

    return {};
  }

 private:
  float eps_{1e-5f};
  int64_t feature_first_{1};  // 1 = [1, features, frames], 0 = [1, frames, features]
};

// NeMo-compatible log-mel spectrogram kernel.
// Wraps nemo_mel::NemoComputeLogMelBatch for use in the SpeechFeatureExtractor pipeline.
// Input:  [num_samples] or [1, num_samples] float32 PCM audio
// Output: [num_mels, num_frames] float32 log-mel spectrogram per example;
//         StackTensors adds the batch dimension later in the pipeline.
class NemoLogMel {
 public:
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "num_mels") {
        cfg_.num_mels = static_cast<int>(std::get<int64_t>(value));
      } else if (key == "fft_size") {
        cfg_.fft_size = static_cast<int>(std::get<int64_t>(value));
      } else if (key == "hop_length") {
        cfg_.hop_length = static_cast<int>(std::get<int64_t>(value));
      } else if (key == "win_length") {
        cfg_.win_length = static_cast<int>(std::get<int64_t>(value));
      } else if (key == "sample_rate") {
        cfg_.sample_rate = static_cast<int>(std::get<int64_t>(value));
      } else if (key == "preemph") {
        cfg_.preemph = static_cast<float>(std::get<double>(value));
      } else if (key == "log_eps") {
        cfg_.log_eps = static_cast<float>(std::get<double>(value));
      } else if (key != "_comment") {
        return {kOrtxErrorInvalidArgument, "[NemoLogMel]: Invalid key in the JSON configuration."};
      }
    }
    return {};
  }

  OrtxStatus Compute(const ortc::Tensor<float>& pcm, ortc::Tensor<float>& logmel) {
    const auto& shape = pcm.Shape();
    size_t num_samples;
    if (shape.size() == 1) {
      num_samples = static_cast<size_t>(shape[0]);
    } else if (shape.size() == 2 && shape[0] == 1) {
      num_samples = static_cast<size_t>(shape[1]);
    } else {
      return {kOrtxErrorInvalidArgument, "[NemoLogMel]: Expected input shape [num_samples] or [1, num_samples]."};
    }

    int num_frames = 0;
    auto mel_data = nemo_mel::NemoComputeLogMelBatch(pcm.Data(), num_samples, cfg_, num_frames);

    // Output [num_mels, num_frames] (no batch dim) — StackTensors adds the batch dim
    auto* out = logmel.Allocate({cfg_.num_mels, num_frames});
    std::memcpy(out, mel_data.data(), mel_data.size() * sizeof(float));
    return {};
  }

 private:
  nemo_mel::NemoMelConfig cfg_{128, 512, 160, 400, 16000, 0.97f, 5.96046448e-08f};
};

}  // namespace ort_extensions
