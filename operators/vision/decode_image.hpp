// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

#include <cstdint>

namespace ort_extensions {
struct KernelDecodeImage : BaseKernel {
  KernelDecodeImage(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {}

  void Compute(OrtKernelContext* context);
};

struct CustomOpDecodeImage : OrtW::CustomOpBase<CustomOpDecodeImage, KernelDecodeImage> {
  void KernelDestroy(void* op_kernel) {
    delete static_cast<KernelDecodeImage*>(op_kernel);
  }

  const char* GetName() const {
    return "DecodeImage";
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
