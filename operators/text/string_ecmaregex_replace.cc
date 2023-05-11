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

void KernelStringECMARegexReplace::Compute(const ortc::Tensor<std::string>& input,
                                           const std::string& pattern,
                                           const std::string& rewrite,
                                           ortc::Tensor<std::string>& output) {
  // make a copy as input is constant;
  std::vector<std::string> str_input = input.Data();
  if (pattern.empty()) {
    ORTX_CXX_API_THROW("pattern (second input) cannot be empty.", ORT_INVALID_GRAPH);
  }
  size_t size = input.NumberOfElement();

  auto regex_flag = std::regex_constants::optimize | std::regex_constants::ECMAScript;
  if (ignore_case_) {
    regex_flag |= std::regex_constants::icase;
  }

  std::regex reg(pattern, regex_flag);

  if (global_replace_) {
    for (size_t i = 0; i < size; i++) {
      str_input[i] = std::regex_replace(str_input[i], reg, rewrite);
    }
  } else {
    for (size_t i = 0; i < size; i++) {
      str_input[i] = std::regex_replace(str_input[i], reg, rewrite, std::regex_constants::format_first_only);
    }
  }

  auto& dimensions = input.Shape();
  output.SetStringOutput(0, str_input, dimensions);
}
