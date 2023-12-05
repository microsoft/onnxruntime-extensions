// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fast_gelu_impl.cuh"
#include <cuda_runtime.h>

template <typename T>
__device__ __inline T _Tanh(T a);

template <>
__device__ __inline__ float _Tanh(float a) { return tanhf(a); }

constexpr float A = 0.5f;

constexpr float B = 0.7978845608028654f;  // sqrt(2.0/M_PI)

constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0/M_PI)

template <typename T, unsigned TPB>
__global__ void FastGeluKernel(const T a, const T b, const T c, int input_length, int bias_length,
                               const T* input, const T* bias, T* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;

  if (idx < input_length) {
    const T x = input[idx];
    const T in = (bias == nullptr) ? x : (T)(x + bias[idx % bias_length]);
    const T cdf = a + a * _Tanh(in * (c * in * in + b));
    output[idx] = in * cdf;
  }
}

template <>
cudaError_t LaunchFastGeluKernel(cudaStream_t stream, int input_length, int bias_length,
                                 const float* input, const float* bias, float* output, bool /*use_half2*/) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  FastGeluKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, input_length, bias_length,
                                                                       input, bias, output);

  return cudaGetLastError();
}