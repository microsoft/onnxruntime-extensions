// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void string_length(const ortc::TensorT<std::string>& input,
                   ortc::TensorT<int64_t>& output);
