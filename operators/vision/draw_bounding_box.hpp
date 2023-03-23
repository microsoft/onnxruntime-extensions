// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"


namespace ort_extensions {
struct DrawBoundingBoxes : BaseKernel {
  DrawBoundingBoxes(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
    thickness_ = TryToGetAttributeWithDefault<int64_t>("thickness", 4);
    num_classes_ = static_cast<int32_t>(TryToGetAttributeWithDefault<int64_t>("num_classes", 10));
    auto mode = TryToGetAttributeWithDefault<std::string>("mode", "xyxy");
    is_xyxy_= mode == "xyxy";
    auto colour_by_classes = TryToGetAttributeWithDefault<int64_t>("colour_by_classes", 1);
    colour_by_classes_ = colour_by_classes > 0;
    if (thickness_<=0){
      ORTX_CXX_API_THROW("[DrawBoundingBoxes] thickness of box should >= 1.", ORT_INVALID_ARGUMENT);
    }
  }

  void Compute(OrtKernelContext* context);

private:
  int64_t thickness_;
  int64_t num_classes_;
  bool colour_by_classes_;
  bool is_xyxy_;
};

struct CustomOpDrawBoundingBoxes : OrtW::CustomOpBase<CustomOpDrawBoundingBoxes, DrawBoundingBoxes> {
  void KernelDestroy(void* op_kernel) {
    delete static_cast<DrawBoundingBoxes*>(op_kernel);
  }

  const char* GetName() const {
    return "DrawBoundingBoxes";
  }

  size_t GetInputTypeCount() const {
    return 2;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    switch (index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      case 1:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
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
