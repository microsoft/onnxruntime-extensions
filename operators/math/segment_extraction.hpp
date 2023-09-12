// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

OrtW::StatusMsg segment_extraction(const ortc::Tensor<int64_t>& input,
                        ortc::Tensor<int64_t>& output0,
                        ortc::Tensor<int64_t>& output1);
