// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_regex_replace.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include "re2/re2.h"
#include "string_tensor.h"

KernelStringRegexReplace::KernelStringRegexReplace(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api, info) {
  global_replace_ = HasAttribute("global_replace") ? ort_.KernelInfoGetAttribute<int64_t>(info_, "global_replace") : 1;
}

void KernelStringRegexReplace::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* pattern = ort_.KernelContext_GetInput(context, 1);
  const OrtValue* rewrite = ort_.KernelContext_GetInput(context, 2);

  std::vector<std::string> str_input, str_pattern, str_rewrite;
  GetTensorMutableDataString(api_, ort_, context, input, str_input);
  GetTensorMutableDataString(api_, ort_, context, pattern, str_pattern);
  GetTensorMutableDataString(api_, ort_, context, rewrite, str_rewrite);

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

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  re2::StringPiece piece(str_rewrite[0]);
  re2::RE2 reg(str_pattern[0]);

  // Do computation
  if (global_replace_) {
    for (int64_t i = 0; i < size; i++) {
      re2::RE2::GlobalReplace(&(str_input[i]), reg, piece);
    }
  } else {
    for (int64_t i = 0; i < size; i++) {
      re2::RE2::Replace(&(str_input[i]), reg, piece);
    }
  }
  FillTensorDataString(api_, ort_, context, str_input, output);
}

void* CustomOpStringRegexReplace::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelStringRegexReplace(api, info);
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
