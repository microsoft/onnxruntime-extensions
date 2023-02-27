// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_upper.hpp"
#include "string_tensor.h"
#include <vector>
#include <cmath>
#include <algorithm>

KernelStringUpper::KernelStringUpper(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
}

void KernelStringUpper::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> X;
  GetTensorMutableDataString(api_, ort_, context, input_X, X);

  for (size_t i = 0; i < X.size(); ++i) {
    std::transform(X[i].begin(), X[i].end(), X[i].begin(), [](char c) { return static_cast<char>(::toupper(c)); });
  }

  // Fills the output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  FillTensorDataString(api_, ort_, context, X, output);
}

const char* CustomOpStringUpper::GetName() const { return "StringUpper"; };

size_t CustomOpStringUpper::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringUpper::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringUpper::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringUpper::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
