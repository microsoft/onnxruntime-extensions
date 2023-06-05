// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void string_length(const ortc::Tensor<std::string>& input,
                   ortc::Tensor<int64_t>& output);
