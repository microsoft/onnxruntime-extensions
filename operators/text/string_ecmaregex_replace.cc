// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_ecmaregex_replace.hpp"
#include <vector>
#include <algorithm>
#include <regex>
#include "string_tensor.h"

OrtStatusPtr KernelStringECMARegexReplace::OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
  auto status = OrtW::GetOpAttribute(info, "global_replace", global_replace_);
  if (!status) {
    status = OrtW::GetOpAttribute(info, "ignore_case", ignore_case_);
  }
  return status;
}

OrtStatusPtr KernelStringECMARegexReplace::Compute(const ortc::Tensor<std::string>& input,
                                           std::string_view pattern,
                                           std::string_view rewrite,
                                           ortc::Tensor<std::string>& output) const {
  // make a copy as input is constant;
  std::vector<std::string> str_input = input.Data();
  if (pattern.empty()) {
    return OrtW::CreateStatus("pattern (second input) cannot be empty.", ORT_INVALID_GRAPH);
  }
  size_t size = input.NumberOfElement();

  auto regex_flag = std::regex_constants::optimize | std::regex_constants::ECMAScript;
  if (ignore_case_) {
    regex_flag |= std::regex_constants::icase;
  }

  std::regex reg(pattern.data(), regex_flag);

  if (global_replace_) {
    for (size_t i = 0; i < size; i++) {
      str_input[i] = std::regex_replace(str_input[i], reg, rewrite.data());
    }
  } else {
    for (size_t i = 0; i < size; i++) {
      str_input[i] = std::regex_replace(str_input[i], reg, rewrite.data(), std::regex_constants::format_first_only);
    }
  }

  auto& dimensions = input.Shape();
  output.SetStringOutput(str_input, dimensions);

  return nullptr;
}
