// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void string_concat(const ortc::TensorT<std::string>& left,
                   const ortc::TensorT<std::string>& right,
                   ortc::TensorT<std::string>& output);
