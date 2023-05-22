// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_regex_replace.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include "re2/re2.h"
#include "string_tensor.h"

KernelStringRegexReplace::KernelStringRegexReplace(const OrtApi& api, const OrtKernelInfo& info)
    : BaseKernel(api, info) {
  global_replace_ = TryToGetAttributeWithDefault("global_replace",1);
}

void KernelStringRegexReplace::Compute(const ortc::Tensor<std::string>& input,
                                        std::string_view str_pattern,
                                        std::string_view str_rewrite,
                                        ortc::Tensor<std::string>& output) {
  if (str_pattern.empty())
    ORTX_CXX_API_THROW("pattern (second input) cannot be empty.", ORT_INVALID_ARGUMENT);

  // Setup output
  std::vector<std::string> str_input{input.Data()};
  auto dim = input.Shape();
  size_t size = input.NumberOfElement();

  re2::StringPiece piece(str_rewrite.data());
  re2::RE2 reg(str_pattern.data());

  if (global_replace_) {
    for (size_t i = 0; i < size; i++) {
      re2::RE2::GlobalReplace(&(str_input[i]), reg, piece);
    }
  } else {
    for (size_t i = 0; i < size; i++) {
      re2::RE2::Replace(&(str_input[i]), reg, piece);
    }
  }
  output.SetStringOutput(str_input, dim);
}