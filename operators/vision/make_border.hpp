// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

#include <array>

namespace ort_extensions {
struct MakeBorder : BaseKernel {
  MakeBorder(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
    mode_ = TryToGetAttributeWithDefault<std::string>("mode", "border_size");  // target_size
    auto fill_value = TryToGetAttributeWithDefault<int64_t>("fill_value", 0);
    fill_value_ = static_cast<uint8_t>(fill_value);
  }

  void Compute(OrtKernelContext* context);

 private:
  uint8_t fill_value_;
  std::string mode_;
};

struct CustomOpMakeBorder : OrtW::CustomOpBase<CustomOpMakeBorder, MakeBorder> {
  void KernelDestroy(void* op_kernel) {
    delete static_cast<MakeBorder*>(op_kernel);
  }

  const char* GetName() const {
    return "MakeBorder";
  }

  size_t GetInputTypeCount() const {
    return 2;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    switch (index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      case 1:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
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
