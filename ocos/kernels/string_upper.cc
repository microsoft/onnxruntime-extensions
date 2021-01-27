// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_upper.hpp"
#include "string_common.h"
#include <vector>
#include <cmath>
#include <algorithm>

KernelStringUpper::KernelStringUpper(OrtApi api) : BaseKernel(api) {
}

void KernelStringUpper::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> X;
  GetTensorMutableDataString(api_, ort_, context, input_X, X);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());

  // Do computation
  for (int64_t i = 0; i < (int64_t)X.size(); ++i) {
    std::transform(X[i].begin(), X[i].end(), X[i].begin(), ::toupper);
  }

  // Fills the output
  FillTensorDataString(api_, ort_, context, X, output);
}

void* CustomOpStringUpper::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelStringUpper(api);
};

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
