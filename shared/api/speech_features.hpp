// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "math/dlib/stft_norm.hpp"

namespace ort_extensions {

class AudioFeatures {
 public:
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "n_fft") {
        n_fft_ = std::get<int64_t>(value);
      } else if (key == "hop_length") {
        hop_length_ = std::get<int64_t>(value);
      } else if (key == "frame_length") {
        frame_length_ = std::get<int64_t>(value);
      } else {
        return {kOrtxErrorInvalidArgument, "[AudioDecoder]: Invalid argument"};
      }
    }
    return {};
  }

  OrtxStatus STFTNorm(const ortc::Tensor<float>& pcm, ortc::Tensor<float>& stft_norm) {
    return stft_norm_.Compute(pcm, n_fft_, hop_length_, {mel_banks_.data(), mel_banks_.size()}, frame_length_,
                              stft_norm);
  }

 private:
  StftNormal stft_norm_;
  int64_t n_fft_{};
  int64_t hop_length_{};
  int64_t frame_length_{};
  std::vector<float> mel_banks_;
};

}  // namespace ort_extensions
