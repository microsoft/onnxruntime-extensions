// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

void segment_sum(const ortc::TensorT<float>& data,
                 const ortc::TensorT<int64_t>& segment_ids,
                 ortc::TensorT<float>& output);
