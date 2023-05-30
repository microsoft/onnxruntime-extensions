// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_strip.hpp"
#include "string_tensor.h"
#include <vector>
#include <cmath>
#include <algorithm>

const char* WHITE_SPACE_CHARS = " \t\n\r\f\v";

KernelStringStrip::KernelStringStrip(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
}

void KernelStringStrip::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> X;
  GetTensorMutableDataString(api_, ort_, context, input_X, X);

  // For each string in input, replace with whitespace-trimmed version.
  for (size_t i = 0; i < X.size(); ++i) {
    size_t nonWhitespaceBegin = X[i].find_first_not_of(WHITE_SPACE_CHARS);
    if (nonWhitespaceBegin != std::string::npos) {
      size_t nonWhitespaceEnd = X[i].find_last_not_of(WHITE_SPACE_CHARS);
      size_t nonWhitespaceRange = nonWhitespaceEnd - nonWhitespaceBegin + 1;

      X[i] = X[i].substr(nonWhitespaceBegin, nonWhitespaceRange);
    }
  }

  // Fills the output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  FillTensorDataString(api_, ort_, context, X, output);
}

const char* CustomOpStringStrip::GetName() const { return "StringStrip"; };

size_t CustomOpStringStrip::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringStrip::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringStrip::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringStrip::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
