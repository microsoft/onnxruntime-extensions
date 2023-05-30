// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_strip.hpp"
#include "string_tensor.h"
#include <vector>
#include <cmath>
#include <algorithm>

const char* WHITE_SPACE_CHARS = " \t\n\r\f\v";

void string_strip(const ortc::Tensor<std::string>& input,
                  ortc::Tensor<std::string>& output) {
  std::vector<std::string> X = input.Data();
  for (size_t i = 0; i < X.size(); ++i) {
    size_t nonWhitespaceBegin = X[i].find_first_not_of(WHITE_SPACE_CHARS);
    if (nonWhitespaceBegin != std::string::npos) {
      size_t nonWhitespaceEnd = X[i].find_last_not_of(WHITE_SPACE_CHARS);
      size_t nonWhitespaceRange = nonWhitespaceEnd - nonWhitespaceBegin + 1;
      X[i] = X[i].substr(nonWhitespaceBegin, nonWhitespaceRange);
    }
  }
  output.SetStringOutput(X, input.Shape());
}
