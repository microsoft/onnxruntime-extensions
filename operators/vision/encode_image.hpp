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
struct KernelEncodeImage : BaseKernel {
  KernelEncodeImage(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel{api, info} {
    OrtW::CustomOpApi op_api{api};
    std::string format = op_api.KernelInfoGetAttribute<std::string>(&info, "format");
    if (format != "jpg" && format != "png") {
      ORTX_CXX_API_THROW("[EncodeImage] 'format' attribute value must be 'jpg' or 'png'.", ORT_RUNTIME_EXCEPTION);
    }

    extension_ = std::string(".") + format;
    encoder_.OnInit();
  }

  void Compute(const ortc::Tensor<uint8_t>& input_bgr, ortc::Tensor<uint8_t>& output) const{
    const auto& dimensions_bgr = input_bgr.Shape();
    if (dimensions_bgr.size() != 3 || dimensions_bgr[2] != 3) {
      ORTX_CXX_API_THROW("[EncodeImage] requires rank 3 BGR input in channels last format.", ORT_INVALID_ARGUMENT);
    }

    int32_t height = static_cast<int32_t>(dimensions_bgr[0]);  // H
    int32_t width = static_cast<int32_t>(dimensions_bgr[1]);   // W
    const int32_t color_space = 3;
    const uint8_t* bgr_data = input_bgr.Data();
    uint8_t* outbuffer = nullptr;
    size_t outsize = 0;

    auto rgb_data = std::make_unique<uint8_t[]>(height * width * color_space);
    for (int32_t y = 0; y < height; ++y) {
      for (int32_t x = 0; x < width; ++x) {
        rgb_data[(y * width + x) * color_space + 0] = bgr_data[(y * width + x) * color_space + 2];
        rgb_data[(y * width + x) * color_space + 1] = bgr_data[(y * width + x) * color_space + 1];
        rgb_data[(y * width + x) * color_space + 2] = bgr_data[(y * width + x) * color_space + 0];
      }
    }

    if (extension_ == ".jpg") {
      encoder_.EncodeJpg(rgb_data.get(), true, width, height, &outbuffer, &outsize);
    } else if (extension_ == ".png") {
      encoder_.EncodePng(rgb_data.get(), true, width, height, &outbuffer, &outsize);
    } else {
      ORTX_CXX_API_THROW("[EncodeImage] Unsupported image format.", ORT_INVALID_ARGUMENT);
    }

    std::vector<int64_t> output_dimensions{static_cast<int64_t>(outsize)};
    uint8_t* data = output.Allocate(output_dimensions);
    memcpy(data, outbuffer, outsize);

    if (outbuffer != nullptr) {
      free(outbuffer);
    }
  }

 private:
  internal::EncodeImage encoder_;
  std::string extension_;
};

}  // namespace ort_extensions
