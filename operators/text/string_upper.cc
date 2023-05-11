// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_upper.hpp"
#include "string_tensor.h"
#include <vector>
#include <cmath>
#include <algorithm>

void string_upper(const ortc::Tensor<std::string>& input,
                  ortc::Tensor<std::string>& output) {
  // Setup inputs
  std::vector<std::string> X = input.Data();

  for (size_t i = 0; i < X.size(); ++i) {
    std::transform(X[i].begin(), X[i].end(), X[i].begin(), [](char c) { return static_cast<char>(::toupper(c)); });
  }

  output.SetStringOutput(0, X, input.Shape());
}
