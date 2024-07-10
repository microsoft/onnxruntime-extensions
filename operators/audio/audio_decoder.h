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
    for (const auto& [key, value] : attrs) {
      if (key == "downsampling_rate") {
        downsample_rate_ = std::get<std::int64_t>(value);
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
  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, const std::optional<std::string> format,
                     ortc::Tensor<float>& output0) const;
  OrtxStatus ComputeNoOpt(const ortc::Tensor<uint8_t>& input, ortc::Tensor<float>& output0) {
    return Compute(input, std::nullopt, output0);
  }

 private:
  int64_t downsample_rate_{};
  int64_t stereo_mixer_{};
};
