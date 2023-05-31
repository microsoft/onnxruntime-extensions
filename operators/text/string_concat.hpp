// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void string_concat(const ortc::Tensor<std::string>& left,
                   const ortc::Tensor<std::string>& right,
                   ortc::Tensor<std::string>& output);
