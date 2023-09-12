// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

OrtW::StatusMsg segment_sum(const ortc::Tensor<float>& data,
                 const ortc::Tensor<int64_t>& segment_ids,
                 ortc::Tensor<float>& output);
