// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void string_split(const ortc::Tensor<std::string>& input_X,
                  std::string_view sep,
                  bool skip_empty,
                  ortc::Tensor<int64_t>& out_indices,
                  ortc::Tensor<std::string>& out_text,
                  ortc::Tensor<int64_t>& out_shape);
