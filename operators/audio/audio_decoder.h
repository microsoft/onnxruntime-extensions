// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

#include <list>
#include <optional>

struct AudioDecoder {
 public:
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info);

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    // in API mode, the default value is 1
    downsample_rates_ = {16000};
    stereo_mixer_ = 1;
    for (const auto& [key, value] : attrs) {
      if (key == "target_sample_rate") {
        downsample_rates_[0] = std::get<std::int64_t>(value);
      } else if (key == "target_sample_rates") {
        downsample_rates_ = std::get<std::vector<std::int64_t>>(value);
        std::sort(downsample_rates_.begin(), downsample_rates_.end());
      } else if (key == "stereo_to_mono") {
        stereo_mixer_ = std::get<std::int64_t>(value);
      } else {
        return {kOrtxErrorInvalidArgument, "[AudioDecoder]: Invalid argument"};
      }
    }

    return {};
  }

  enum class AudioStreamType { kDefault = 0, kWAV, kMP3, kFLAC };

  AudioStreamType ReadStreamFormat(const uint8_t* p_data, const std::string& str_format, OrtxStatus& status) const;

  OrtxStatus ComputeInternal(const ortc::Tensor<uint8_t>& input,
                             const std::optional<std::string> format,
                             ortc::Tensor<float>& pcm, int64_t& sr) const;

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, const std::optional<std::string> format,
                     ortc::Tensor<float>& output0) const {
    int64_t sr{};
    return ComputeInternal(input, format, output0, sr); }

  OrtxStatus ComputeNoOpt(const ortc::Tensor<uint8_t>& input, ortc::Tensor<float>& pcm) const {
    int64_t sr{};
    return ComputeInternal(input, std::nullopt, pcm, sr); }

  OrtxStatus ComputeNoOpt2(const ortc::Tensor<uint8_t>& input,
                           ortc::Tensor<float>& pcm,
                           ortc::Tensor<int64_t>& sr) const;

 private:
  int64_t stereo_mixer_{};
  std::vector<int64_t> downsample_rates_{};
};
