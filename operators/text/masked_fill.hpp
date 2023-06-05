// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"
#include <unordered_map>

void masked_fill(const ortc::Tensor<std::string>& input,
                 const ortc::Tensor<bool>& input_mask,
                 ortc::Tensor<std::string>& output);
