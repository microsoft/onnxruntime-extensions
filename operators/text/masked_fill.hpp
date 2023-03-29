// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"
#include <unordered_map>

void masked_fill(const ortc::TensorT<std::string>& input,
                 const ortc::TensorT<bool>& input_mask,
                 ortc::TensorT<std::string>& output);
