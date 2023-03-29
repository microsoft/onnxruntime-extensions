// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_concat.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>

void string_concat(const ortc::TensorT<std::string>& left,
                   const ortc::TensorT<std::string>& right,
                   ortc::TensorT<std::string>& output) {
  if (left.Shape() != right.Shape()) {
    ORTX_CXX_API_THROW("Two input tensor should have the same dimension.", ORT_INVALID_ARGUMENT);
  }
  // make a copy as input is const
  std::vector<std::string> left_value = left.Data();
  auto& right_value = right.Data();

  for (size_t i = 0; i < left_value.size(); i++) {
    left_value[i].append(right_value[i]);
  }

  output.SetStringOutput(0, left_value, left.Shape());
}
