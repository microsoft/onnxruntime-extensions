// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "sampling.h"

#include <list>
#include <optional>

struct AudioDecoder {
 public:
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    auto status = OrtW::GetOpAttribute(info, "downsampling_rate", downsample_rate_);
    if (!status) {
      status = OrtW::GetOpAttribute(info, "stereo_to_mono", stereo_mixer_);
    }
  }

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    // in API mode, the default value is 1
    downsample_rate_ = 16000;
    stereo_mixer_ = 1;
    for (const auto& [key, value] : attrs) {
      if (key == "target_sample_rate") {
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

  void MixAndDownsampleIfNeeded(std::vector<float>& buf, int64_t orig_channels, int64_t orig_sample_rate) const {
    // mix the stereo channels into mono channel
    if (stereo_mixer_ && orig_channels > 1) {
      if (buf.size() > 1) {
        for (size_t i = 0; i < buf.size() / 2; ++i) {
          buf[i] = (buf[i * 2] + buf[i * 2 + 1]) / 2;
        }
        buf.resize(buf.size() / 2);
      }
    }

    if (downsample_rate_ != 0 && downsample_rate_ != orig_sample_rate) {
      // A lowpass filter on buf audio data to remove high frequency noise
      ButterworthLowpass filter(0.5 * downsample_rate_, 1.0 * orig_sample_rate);
      std::vector<float> filtered_buf = filter.Process(buf);
      // downsample the audio data
      KaiserWindowInterpolation::Process(filtered_buf, buf, 1.0f * orig_sample_rate, 1.0f * downsample_rate_);
    }
  }

 private:
  int64_t downsample_rate_{};
  int64_t stereo_mixer_{};
};
