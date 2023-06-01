// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

#include <cstdint>

namespace ort_extensions {
struct KernelEncodeImage : BaseKernel {
  KernelEncodeImage(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel{api, info} {
    OrtW::CustomOpApi op_api{api};
    std::string format = op_api.KernelInfoGetAttribute<std::string>(&info, "format");
    if (format != "jpg" && format != "png") {
      ORTX_CXX_API_THROW("[EncodeImage] 'format' attribute value must be 'jpg' or 'png'.", ORT_RUNTIME_EXCEPTION);
    }

    extension_ = std::string(".") + format;
  }

  void Compute(const ortc::Tensor<uint8_t>& input_bgr,
               ortc::Tensor<uint8_t>& output);

 private:
  std::string extension_;
};

}  // namespace ort_extensions
