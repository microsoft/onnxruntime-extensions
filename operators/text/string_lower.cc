// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_lower.hpp"
#include "string_tensor.h"
#include "ustring.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iterator>

void string_lower(const ortc::Tensor<std::string>& input,
                  ortc::Tensor<std::string>& output) {
  const auto& input_strings = input.Data();

  std::vector<std::string> output_strings;
  output_strings.reserve(input_strings.size());

  std::transform(input_strings.begin(), input_strings.end(), std::back_inserter(output_strings),
                 [](const std::string& input_string) {
                   ustring u32_input_string(input_string);
                   std::transform(u32_input_string.begin(), u32_input_string.end(), u32_input_string.begin(),
                                  [](char32_t c) { return ToLower(c); });
                   return static_cast<std::string>(u32_input_string);
                 });

  output.SetStringOutput(output_strings, input.Shape());
}
