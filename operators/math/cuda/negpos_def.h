// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"

OrtStatusPtr neg_pos_cuda(const Ort::Custom::CudaContext& ctx,
                          const ortc::Tensor<float>& input,
                          ortc::Tensor<float>& out0_tensor,
                          ortc::Tensor<float>& out1_tensor);
