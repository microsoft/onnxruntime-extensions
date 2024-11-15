// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstring>
#include <variant>
#include <unordered_map>

#include "ext_status.h"
#include "op_def_struct.h"

#if OCOS_ENABLE_VENDOR_IMAGE_CODECS
  #if _WIN32
    #include "image_decoder_win32.hpp"
  #elif __APPLE__
    #include "image_decoder_darwin.hpp"
  #else
    #include "image_decoder.hpp"
  #endif
#else
  #include "image_decoder.hpp"
#endif

namespace ort_extensions {
struct DecodeImage: public internal::DecodeImage {

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    auto status = internal::DecodeImage::OnInit();
    if (!status.IsOk()) {
      return status;
    }

    for (const auto& [key, value] : attrs) {
      if (key == "color_space") {
        auto color_space = std::get<std::string>(value);
        if (color_space == "RGB") {
          is_bgr_ = false;
        } else if (color_space == "BGR") {
          is_bgr_ = true;
        } else {
          return {kOrtxErrorInvalidArgument, "[DecodeImage]: Invalid color_space"};
        }
      } else {
        return {kOrtxErrorInvalidArgument, "[Resize]: Invalid argument"};
      }
    }

    return {};
  }

  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    is_bgr_ = true;
    return Init(std::unordered_map<std::string, std::variant<std::string>>());
  }

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) const{
    auto status = internal::DecodeImage::Compute(input, output);
    if (!status.IsOk()) {
      return status;
    }

    if (is_bgr_) {
      // need to convert rgb to bgr for backward compatibility
      const auto& dimensions = output.Shape();
      uint8_t* rgb_data = const_cast<uint8_t*>(output.Data());
      // do an inplace swap of the channels
      for (int y = 0; y < dimensions[0]; ++y) {
        for (int x = 0; x < dimensions[1]; ++x) {
          std::swap(rgb_data[(y * dimensions[1] + x) * 3 + 0], rgb_data[(y * dimensions[1] + x) * 3 + 2]);
        }
      }
    }

    return status;
  }

  private:
    bool is_bgr_{};  // flag to indicate if the output is in BGR format
};

}  // namespace ort_extensions
