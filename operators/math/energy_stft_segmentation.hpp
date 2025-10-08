// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

// It splits the input signal into high-energy segments separated by gaps of low-energy segments using STFT.
OrtStatusPtr split_signal_energy_segments(const ortc::Tensor<float>& input, 
                                          const ortc::Tensor<int64_t>& sr_tensor,
                                          const ortc::Tensor<int64_t>& frame_ms_tensor,
                                          const ortc::Tensor<int64_t>& hop_ms_tensor,
                                          const ortc::Tensor<float>& energy_threshold_db_tensor,
                                          ortc::Tensor<int64_t>& output0);

// Merge signal segments that are distanced by less than the specified merge gap.
OrtStatusPtr merge_signal_segments(const ortc::Tensor<int64_t>& segments_tensor,
                                   const ortc::Tensor<int64_t>& merge_gap_ms_tensor, ortc::Tensor<int64_t>& output0);