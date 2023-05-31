// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void string_strip(const ortc::Tensor<std::string>& input,
                  ortc::Tensor<std::string>& output);