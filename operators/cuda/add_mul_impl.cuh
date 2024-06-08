// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
cudaError_t LaunchAddOrMulSharedInputKernel(cudaStream_t stream, const T* input_a, const T* input_b, const T* input_c,
                                            T* output_ab, T* output_ac,
                                            int64_t length_a, int64_t length_b, int64_t length_c, bool addition);

template <typename T>
cudaError_t LaunchAddOrMulTwiceKernel(cudaStream_t stream, const T* input_a, const T* input_b, const T* input_c,
                                      T* output, int64_t length_a, int64_t length_b, int64_t length_c, bool addition);

template <typename T>
cudaError_t LaunchAddAndMulKernel(cudaStream_t stream, const T* input_a, const T* input_b, const T* input_c,
                                  T* output, int64_t length_a, int64_t length_b, int64_t length_c,
                                  bool addition, bool switchMiddleAxis);

template <typename T>
cudaError_t LaunchAddAndMulSwitchMiddleAxesKernel(cudaStream_t stream, const T* input_a, const T* input_b, const T* input_c,
                                                  T* output, int64_t length_a, int64_t length_b, int64_t length_c,
                                                  bool addition,
                                                  int64_t d2, int64_t d3, int64_t d4);
