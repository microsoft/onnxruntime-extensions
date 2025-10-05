// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

OrtStatusPtr segment_extraction(const ortc::Tensor<int64_t>& input,
                        ortc::Tensor<int64_t>& output0,
                        ortc::Tensor<int64_t>& output1);

OrtStatusPtr segment_extraction2(const ortc::Tensor<float>& audio,
 const ortc::Tensor<int64_t>& sr_tensor,
    const ortc::Tensor<int64_t>& frame_ms_tensor,
    const ortc::Tensor<int64_t>& hop_ms_tensor,
    const ortc::Tensor<float>& energy_threshold_db_tensor,
                                 ortc::Tensor<int64_t>& output0);