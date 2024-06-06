// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

enum class RotarySide : int {
  LEFT = 1,
  RIGHT = 2,
};

template <typename T>
cudaError_t LaunchRotaryKernel(cudaStream_t stream, int input_length, int last_dim,
                               const T* input, const int64_t* split_data, T* output, RotarySide side);
