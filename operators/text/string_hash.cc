// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include "string_tensor.h"
#include "string_functions.h"

OrtStatusPtr string_hash(const ortc::Tensor<std::string>& input,
                 int64_t num_buckets,
                 ortc::Tensor<int64_t>& output) {
  // Setup inputs
  auto& str_input = input.Data();

  // Setup output
  auto& dimensions = input.Shape();
  int64_t* out = output.Allocate(dimensions);

  size_t size = output.NumberOfElement();

  // Do computation
  size_t nb = static_cast<size_t>(num_buckets);
  for (size_t i = 0; i < size; i++) {
    out[i] = static_cast<int64_t>(Hash64(str_input[i].c_str(), str_input[i].size()) % nb);
  }

  return nullptr;
}

OrtStatusPtr string_hash_fast(const ortc::Tensor<std::string>& input,
                               int64_t num_buckets,
                               ortc::Tensor<int64_t>& output) {
  auto& str_input = input.Data();
  auto& dimensions = input.Shape();
  int64_t* out = output.Allocate(dimensions);
  size_t size = output.NumberOfElement();

  std::hash<std::string> hasher;
  size_t nb = static_cast<size_t>(num_buckets);
  for (size_t i = 0; i < size; i++) {
    out[i] = static_cast<int64_t>(hasher(str_input[i]) % nb);
  }

  return nullptr;
}
