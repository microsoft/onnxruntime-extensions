// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_concat.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>

KernelStringConcat::KernelStringConcat(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
}

void KernelStringConcat::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* left = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* right = ort_.KernelContext_GetInput(context, 1);
  OrtTensorDimensions left_dim(ort_, left);
  OrtTensorDimensions right_dim(ort_, right);

  if (left_dim != right_dim) {
    ORTX_CXX_API_THROW("Two input tensor should have the same dimension.", ORT_INVALID_ARGUMENT);
  }

  std::vector<std::string> left_value;
  std::vector<std::string> right_value;
  GetTensorMutableDataString(api_, ort_, context, left, left_value);
  GetTensorMutableDataString(api_, ort_, context, right, right_value);

  // reuse left_value as output to save memory
  for (size_t i = 0; i < left_value.size(); i++) {
    left_value[i].append(right_value[i]);
  }

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, left_dim.data(), left_dim.size());
  FillTensorDataString(api_, ort_, context, left_value, output);
}

const char* CustomOpStringConcat::GetName() const { return "StringConcat"; };

size_t CustomOpStringConcat::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpStringConcat::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringConcat::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringConcat::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
