// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "masked_fill.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>

void masked_fill(const ortc::Tensor<std::string>& input,
                 const ortc::Tensor<bool>& input_mask,
                 ortc::Tensor<std::string>& output) {
  auto& value_dimensions = input.Shape();
  auto& mask_dimensions = input_mask.Shape();
  if (!(value_dimensions.empty() || mask_dimensions.size() == 1)) {
    ORTX_CXX_API_THROW("[MaskedFill]: the dimension of input value should be vector or scalar.", ORT_INVALID_ARGUMENT);
  }

  if (value_dimensions != mask_dimensions) {
    ORTX_CXX_API_THROW("[MaskedFill]: the dimension of input value and mask should be same.", ORT_INVALID_ARGUMENT);
  }

  auto& value = input.Data();
  const bool* mask = input_mask.Data();

  std::vector<std::string> result;
  std::vector<int64_t> result_dimension;

  for (size_t i = 0; i < value.size(); i++) {
    if (!mask[i]) {
      continue;
    }

    result.push_back(value[i]);
  }
  result_dimension.push_back(result.size());
  output.SetStringOutput(0, result, result_dimension);
}
