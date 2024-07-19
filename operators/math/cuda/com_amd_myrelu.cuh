// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

void com_amd_myrelu_impl(cudaStream_t stream,
                         const float* input, float* out, int size);
