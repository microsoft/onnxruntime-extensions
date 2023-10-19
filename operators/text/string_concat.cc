// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_functions.h"
#include "string_tensor.h"
#include <vector>
#include <algorithm>

OrtStatusPtr string_concat(const ortc::Tensor<std::string>& left,
                   const ortc::Tensor<std::string>& right,
                   ortc::Tensor<std::string>& output) {
  OrtStatusPtr status = nullptr;
  if (left.Shape() != right.Shape()) {
    status = OrtW::CreateStatus("Two input tensor should have the same dimension.", ORT_INVALID_ARGUMENT);
    return status;
  }
  // make a copy as input is const
  std::vector<std::string> left_value = left.Data();
  auto& right_value = right.Data();

  for (size_t i = 0; i < left_value.size(); i++) {
    left_value[i].append(right_value[i]);
  }

  output.SetStringOutput(left_value, left.Shape());
  return status;
}
