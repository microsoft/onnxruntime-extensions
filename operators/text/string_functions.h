// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"


OrtStatusPtr string_concat(const ortc::Tensor<std::string>& left,
                   const ortc::Tensor<std::string>& right,
                   ortc::Tensor<std::string>& output);

OrtStatusPtr string_join(const ortc::Tensor<std::string>& input_X,
                 std::string_view input_sep,
                 int64_t axis,
                 ortc::Tensor<std::string>& output);

OrtStatusPtr string_length(const ortc::Tensor<std::string>& input,
                   ortc::Tensor<int64_t>& output);

OrtStatusPtr string_lower(const ortc::Tensor<std::string>& input,
                  ortc::Tensor<std::string>& output);

OrtStatusPtr string_upper(const ortc::Tensor<std::string>& input,
                  ortc::Tensor<std::string>& output);

OrtStatusPtr string_split(const ortc::Tensor<std::string>& input_X,
                  std::string_view sep,
                  bool skip_empty,
                  ortc::Tensor<int64_t>& out_indices,
                  ortc::Tensor<std::string>& out_text,
                  ortc::Tensor<int64_t>& out_shape);

OrtStatusPtr string_strip(const ortc::Tensor<std::string>& input,
                  ortc::Tensor<std::string>& output);

OrtStatusPtr string_equal(const ortc::Tensor<std::string>& input_1,
                                const ortc::Tensor<std::string>& input_2,
                                ortc::Tensor<bool>& output);

OrtStatusPtr string_hash(const ortc::Tensor<std::string>& input,
                 int64_t num_buckets,
                 ortc::Tensor<int64_t>& output);

OrtStatusPtr string_hash_fast(const ortc::Tensor<std::string>& input,
                      int64_t num_buckets,
                      ortc::Tensor<int64_t>& output);

OrtStatusPtr masked_fill(const ortc::Tensor<std::string>& input,
                 const ortc::Tensor<bool>& input_mask,
                 ortc::Tensor<std::string>& output);
