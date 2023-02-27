// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_ecmaregex_replace.hpp"
#include <vector>
#include <algorithm>
#include <regex>
#include "string_tensor.h"

KernelStringECMARegexReplace::KernelStringECMARegexReplace(const OrtApi& api, const OrtKernelInfo& info)
    : BaseKernel(api, info) {
  global_replace_ = TryToGetAttributeWithDefault("global_replace", true);
  ignore_case_ = TryToGetAttributeWithDefault("ignore_case", false);
}

void KernelStringECMARegexReplace::Compute(OrtKernelContext* context) {
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
  if (pattern_dimensions.Size() != 1) {
    ORTX_CXX_API_THROW(MakeString("pattern (second input) must contain only one element. It has ",
                                  pattern_dimensions.size(), " dimensions."),
                       ORT_INVALID_GRAPH);
  }
  if (rewrite_dimensions.Size() != 1) {
    ORTX_CXX_API_THROW(MakeString("rewrite (third input) must contain only one element. It has ",
                                  rewrite_dimensions.size(), " dimensions."),
                       ORT_INVALID_GRAPH);
  }
  if (str_pattern[0].empty()) {
    ORTX_CXX_API_THROW("pattern (second input) cannot be empty.", ORT_INVALID_GRAPH);
  }

  // Setup output
  OrtTensorDimensions dimensions(ort_, input);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  size_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  auto regex_flag = std::regex_constants::optimize | std::regex_constants::ECMAScript;
  if (ignore_case_) {
    regex_flag |= std::regex_constants::icase;
  }

  std::regex reg(str_pattern[0], regex_flag);

  if (global_replace_) {
    for (size_t i = 0; i < size; i++) {
      str_input[i] = std::regex_replace(str_input[i], reg, str_rewrite[0]);
    }
  } else {
    for (size_t i = 0; i < size; i++) {
      str_input[i] = std::regex_replace(str_input[i], reg, str_rewrite[0], std::regex_constants::format_first_only);
    }
  }

  FillTensorDataString(api_, ort_, context, str_input, output);
}

const char* CustomOpStringECMARegexReplace::GetName() const { return "StringECMARegexReplace"; };

size_t CustomOpStringECMARegexReplace::GetInputTypeCount() const {
  return 3;
};

ONNXTensorElementDataType CustomOpStringECMARegexReplace::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringECMARegexReplace::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringECMARegexReplace::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
