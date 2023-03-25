// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void segment_extraction(const ortc::TensorT<int64_t>& input,
                        ortc::TensorT<int64_t>& output0,
                        ortc::TensorT<int64_t>& output1);
