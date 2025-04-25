// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

#include <cstdint>

#if OCOS_ENABLE_VENDOR_IMAGE_CODECS
#if _WIN32
#include "image_encoder_win32.hpp"
#elif __APPLE__
#include "image_encoder_darwin.hpp"
#else
#include "image_encoder.hpp"
#endif
#else
#include "image_encoder.hpp"
#endif

namespace ort_extensions {
struct EncodeImage: public internal::EncodeImage {

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    auto status = internal::EncodeImage::OnInit();
    if (!status.IsOk()) {
      return status;
    }

    for (const auto& [key, value] : attrs) {
      if (key == "color_space") {
        auto color_space = std::get<std::string>(value);
        std::transform(color_space.begin(), color_space.end(), color_space.begin(), ::toupper);
        if (color_space == "RGB") {
          is_bgr_ = false;
        } else if (color_space == "BGR") {
          is_bgr_ = true;
        } else {
          return {kOrtxErrorInvalidArgument, "[EncodeImage]: Invalid color_space"};
        }
      } else if (key == "file_extension") {
        extension_ = std::get<std::string>(value);
        std::transform(extension_.begin(), extension_.end(), extension_.begin(), ::tolower);
        if (extension_ != ".jpg" && extension_ != ".png") {
          return {kOrtxErrorInvalidArgument, "[EncodeImage]: Invalid format"};
        }
      } else {
        return {kOrtxErrorInvalidArgument, "[EncodeImage]: Invalid argument"};
      }
    }

    return {};
  }

  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    std::unordered_map<std::string, std::variant<std::string>> attrs = {
        {"color_space", "bgr"},
        {"file_extension", "png"}
    };

    std::string format;
    auto status = OrtW::API::GetOpAttributeString(api, info, "format", format);
    if (status != nullptr) {
      return status;
    }
    if (!format.empty()){
      if (format != "jpg" && format != "png") {
        return OrtW::CreateStatus("[EncodeImage] 'format' attribute value must be 'jpg' or 'png'.", ORT_RUNTIME_EXCEPTION);
      } else {
        attrs["file_extension"] = std::string(".") + format;
      }
    }

    std::string clr;
    status = OrtW::API::GetOpAttributeString(api, info, "color_space", clr);
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
    const auto& dimensions = input.Shape();
    if (dimensions.size() != 3 || dimensions[2] != 3) {
      return {kOrtxErrorInvalidArgument, "[EncodeImage] requires rank 3 rgb input in channels last format."};
    }

    int32_t height = static_cast<int32_t>(dimensions[0]);  // H
    int32_t width = static_cast<int32_t>(dimensions[1]);   // W
    const int32_t color_space = 3;
    const uint8_t* input_data_ptr = input.Data();
    uint8_t* outbuffer = nullptr;
    size_t outsize = 0;

    std::unique_ptr<uint8_t[]> conversion_buf;
    auto encoding_data = input_data_ptr;
    bool conversion_needed = false;
    auto fx_supported_bgr =
      extension_ == ".png"? &internal::EncodeImage::pngSupportsBgr : &internal::EncodeImage::JpgSupportsBgr;

    bool bgr_source = is_bgr_;
    if ((this->*fx_supported_bgr)() != is_bgr_) {
      conversion_needed = true;
      bgr_source = !is_bgr_;
    }

    if (conversion_needed) {
      conversion_buf = std::make_unique<uint8_t[]>(height * width * color_space);
      auto cvt_data = conversion_buf.get();
      for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
          cvt_data[(y * width + x) * color_space + 0] = input_data_ptr[(y * width + x) * color_space + 2];
          cvt_data[(y * width + x) * color_space + 1] = input_data_ptr[(y * width + x) * color_space + 1];
          cvt_data[(y * width + x) * color_space + 2] = input_data_ptr[(y * width + x) * color_space + 0];
        }
      }
      encoding_data = cvt_data;
    }

    OrtxStatus status{};

    if (extension_ == ".jpg") {
      status = EncodeJpg(encoding_data, bgr_source, width, height, &outbuffer, &outsize);
    } else if (extension_ == ".png") {
      status = EncodePng(encoding_data, bgr_source, width, height, &outbuffer, &outsize);
    } else {
      status = {kOrtxErrorInvalidArgument, "[EncodeImage] Unsupported image format."};
    }

    if (!status.IsOk()) {
      return status;
    }

    std::vector<int64_t> output_dimensions{static_cast<int64_t>(outsize)};
    uint8_t* data = output.Allocate(output_dimensions);
    memcpy(data, outbuffer, outsize);

    if (outbuffer != nullptr) {
      free(outbuffer);
    }

    return {};
  }

 private:
  std::string extension_{".png"};
  bool is_bgr_{};   // is the input data buffer in BGR format?
};

}  // namespace ort_extensions
