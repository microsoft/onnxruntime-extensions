// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_regex_replace.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include "re2/re2.h"

KernelStringRegexReplace::KernelStringRegexReplace(OrtApi api) : BaseKernel(api) {
}

void KernelStringRegexReplace::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  const std::string* str_input = ort_.GetTensorData<std::string>(input);
  const OrtValue* pattern = ort_.KernelContext_GetInput(context, 1);
  const std::string* str_pattern = ort_.GetTensorData<std::string>(pattern);
  const OrtValue* rewrite = ort_.KernelContext_GetInput(context, 2);
  const std::string* str_rewrite = ort_.GetTensorData<std::string>(rewrite);

  // Verifications
  OrtTensorDimensions pattern_dimensions(ort_, pattern);
  OrtTensorDimensions rewrite_dimensions(ort_, rewrite);
  if (pattern_dimensions.size() != 1 || pattern_dimensions[0] != 1)
    throw std::runtime_error(MakeString(
        "pattern (second input) must contain only one element. It has ",
        pattern_dimensions.size(), " dimensions."));
  if (rewrite_dimensions.size() != 1 || rewrite_dimensions[0] != 1)
    throw std::runtime_error(MakeString(
        "rewrite (third input) must contain only one element. It has ",
        rewrite_dimensions.size(), " dimensions."));

  // Setup output
  OrtTensorDimensions dimensions(ort_, input);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  std::string* out = ort_.GetTensorMutableData<std::string>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  re2::StringPiece piece(*str_rewrite);
  re2::RE2 reg(*str_pattern);

  // Do computation
  for (int64_t i = 0; i < size; i++) {
    out[i] = str_input[i];
    re2::RE2::GlobalReplace(out + i, reg, piece);
  }
}

void* CustomOpStringRegexReplace::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
  return new KernelStringRegexReplace(api);
};

const char* CustomOpStringRegexReplace::GetName() const { return "StringRegexReplace"; };

size_t CustomOpStringRegexReplace::GetInputTypeCount() const {
  return 3;
};

ONNXTensorElementDataType CustomOpStringRegexReplace::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringRegexReplace::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringRegexReplace::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
