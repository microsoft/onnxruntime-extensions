// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void string_join(const ortc::Tensor<std::string>& input_X,
                 std::string_view input_sep,
                 int64_t axis,
                 ortc::Tensor<std::string>& output);
