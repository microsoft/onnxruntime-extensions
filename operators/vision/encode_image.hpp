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

  void Compute(OrtKernelContext* context);

 private:
  std::string extension_;
};

/// <summary>
/// EncodeImage
///
/// Converts rank 3 BGR input with channels last ordering to the requested file type.
/// Default is 'jpg'
/// </summary>
struct CustomOpEncodeImage : OrtW::CustomOpBase<CustomOpEncodeImage, KernelEncodeImage> {
  const char* GetName() const {
    return "EncodeImage";
  }

  size_t GetInputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    switch (index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      default:
        ORTX_CXX_API_THROW(MakeString("Invalid input index ", index), ORT_INVALID_ARGUMENT);
    }
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    switch (index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      default:
        ORTX_CXX_API_THROW(MakeString("Invalid output index ", index), ORT_INVALID_ARGUMENT);
    }
  }
};
}  // namespace ort_extensions
