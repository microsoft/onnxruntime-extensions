// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstring>
#include <variant>
#include <unordered_map>

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
        std::string color_space = std::get<std::string>(value);
        std::transform(color_space.begin(), color_space.end(), color_space.begin(), ::toupper);
        if (color_space == "RGB") {
          is_bgr_ = false;
        } else if (color_space == "BGR") {
          is_bgr_ = true;
        } else {
          return {kOrtxErrorInvalidArgument, "[DecodeImage]: Invalid color_space"};
        }
      } else {
        return {kOrtxErrorInvalidArgument, "[DecodeImage]: Invalid argument"};
      }
    }

    return {};
  }

  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    std::unordered_map<std::string, std::variant<std::string>> attrs = {
        {"color_space", "bgr"}
    };

    std::string clr;
    auto status = OrtW::API::GetOpAttributeString(api, info, "color_space", clr);
    if (status != nullptr) {
      return status;
    }
    if (!clr.empty()) {
      if (clr != "bgr" && clr != "rgb") {
        return OrtW::CreateStatus("[EncodeImage] 'color_space' attribute value must be 'bgr' or 'rgb'.", ORT_RUNTIME_EXCEPTION);
      } else {
        attrs["color_space"] = clr;
      }
    }

    return Init(attrs);
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
    std::string image_type_{"png"};
    bool is_bgr_{};  // flag to indicate if the output is in BGR format
};

}  // namespace ort_extensions
