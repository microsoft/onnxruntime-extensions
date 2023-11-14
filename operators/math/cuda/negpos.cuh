// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

void neg_pos_impl(cudaStream_t stream,
                  const float* input, float* pos_out, float* neg_out, int size);
