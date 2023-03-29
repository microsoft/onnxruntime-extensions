// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void string_split(const ortc::TensorT<std::string>& input_X,
                  const std::string& sep,
                  bool skip_empty,
                  ortc::TensorT<int64_t>& out_indices,
                  ortc::TensorT<std::string>& out_text,
                  ortc::TensorT<int64_t>& out_shape);
