// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void string_join(const ortc::TensorT<std::string>& input_X,
                 const std::string& input_sep,
                 int64_t axis,
                 ortc::TensorT<std::string>& output);
