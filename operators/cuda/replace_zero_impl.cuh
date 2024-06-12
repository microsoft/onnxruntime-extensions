// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
cudaError_t LaunchReplaceZeroKernel(cudaStream_t stream, int input_length, const T* input_data, T* output_data, float by);
