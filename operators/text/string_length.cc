// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_length.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>
#include "ustring.h"

void string_length(const ortc::TensorT<std::string>& input,
                   ortc::TensorT<int64_t>& output) {
  // Setup inputs
  auto& input_data = input.Data();

  auto& dimensions = input.Shape();
  auto* output_data = output.Allocate(dimensions);

  for (int i = 0; i < input.NumberOfElement(); i++) {
    output_data[i] = ustring(input_data[i]).size();
  }
}
