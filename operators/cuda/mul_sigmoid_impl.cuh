// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
cudaError_t LaunchMulSigmoidKernel(cudaStream_t stream, int input_length, const T* input, T* output);

template <typename T>
cudaError_t LaunchMulMulSigmoidKernel(cudaStream_t stream, int input_length_x, int input_length_y,
                                      const T* input_data_x, const T* input_data_y, T* output);
