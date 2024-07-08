// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "math/dlib/stft_norm.hpp"

namespace ort_extensions {

class SpeechFeatures {
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
      } else if (key == "hann_win") {
        auto& win = std::get<std::vector<double>>(value);
        hann_win_.resize(win.size());
        std::transform(win.begin(), win.end(), hann_win_.begin(), [](double x) { return static_cast<float>(x); });
      } else {
        return {kOrtxErrorInvalidArgument, "[AudioFeatures]: Invalid key in the JSON configuration."};
      }
    }
    return {};
  }

  OrtxStatus STFTNorm(const ortc::Tensor<float>& pcm, ortc::Tensor<float>& stft_norm) {
    return stft_norm_.Compute(pcm, n_fft_, hop_length_, {hann_win_.data(), hann_win_.size()}, frame_length_, stft_norm);
  }

 private:
  StftNormal stft_norm_;
  int64_t n_fft_{};
  int64_t hop_length_{};
  int64_t frame_length_{};
  std::vector<float> hann_win_;
};

class LogMel {
 public:
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "hop_length") {
        hop_length_ = std::get<int64_t>(value);
      } else if (key == "n_samples") {
        n_samples_ = std::get<int64_t>(value);
      } else if (key == "mel_filters_index") {
        mel_filters_index_ = std::get<std::vector<int64_t>>(value);
      } else if (key == "mel_filters") {
        auto& filters = std::get<std::vector<double>>(value);
        mel_filters_.resize(filters.size());
        std::transform(filters.begin(), filters.end(), mel_filters_.begin(),
                       [](double x) { return static_cast<float>(x); });
      } else {
        return {kOrtxErrorInvalidArgument, "[LogMel]: Invalid key in the JSON configuration."};
      }
    }
    return {};
  }

  OrtxStatus Compute(const ortc::Tensor<float>& stft_norm, ortc::Tensor<float>& logmel) {
    // magnitudes = stft_norm[:, :, :-1]
    // mel_spec = self.mel_filters @ magnitudes
    // log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    // spec_min = log_spec.max() - 8.0
    // log_spec = torch.maximum(log_spec, spec_min)
    // spec_shape = log_spec.shape
    // padding_spec = torch.ones(spec_shape[0],
    //                           spec_shape[1],
    //                           self.n_samples // self.hop_length - spec_shape[2],
    //                           dtype=torch.float)
    // padding_spec *= spec_min
    // log_spec = torch.cat((log_spec, padding_spec), dim=2)
    // log_spec = (log_spec + 4.0) / 4.0
    // return log_spec
    assert(stft_norm.Shape().size() == 3 && stft_norm.Shape()[0] == 1);
    std::vector<int64_t> shape = {stft_norm.Shape()[1], stft_norm.Shape()[2], n_samples_ / hop_length_};
    float* buff = logmel.Allocate(shape);
    memcpy(buff, stft_norm.Data(), stft_norm.NumberOfElement() * sizeof(float));
    return {};
  }

 private:
  int64_t n_samples_ = 16000 * 30;  // sr * chunk_size
  int64_t hop_length_{};
  std::vector<float> mel_filters_;
  std::vector<int64_t> mel_filters_index_;
};

}  // namespace ort_extensions
