// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"

OrtStatusPtr com_amd_myrelu_cuda(Ort::Custom::CUDAKernelContext* ctx,
                                 const ortc::Tensor<float>& input,
                                 ortc::Tensor<float>& out0_tensor);
