// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

void launch_cuda_kernel(cudaStream_t stream,
                        int64_t input_size,
                        int64_t bias_size,
                        const float* X,
                        const float* B,
                        float* Y);