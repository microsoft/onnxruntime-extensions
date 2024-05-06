// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_prop.cuh"
#include "utils.cuh"
#include "negxplus1_impl.cuh"
#include "cuda_type.h"

using namespace Ort::Custom;

template <typename T>
__device__ __inline__ T _negxplus1(const T x) {
  return (T)1 - x;
}

template <>
__device__ __inline__ half _negxplus1(const half x) {
#if __CUDA_ARCH__ < 700
  return __float2half(1 - __half2float(x));
#else
  return (half)1 - x;
#endif
}

template <typename T>
__global__ void NegXPlus1Kernel(T* output_data, const T* input_data, int N) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= N)
    return;
  output_data[id] = _negxplus1(input_data[id]);
}

template <typename T>
cudaError_t _LaunchNegXPlus1Kernel(cudaStream_t stream, int input_length, const T* input, T* output) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  using TT = typename contrib::CudaT<T>::MappedType;
  NegXPlus1Kernel<TT><<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<TT*>(output), reinterpret_cast<const TT*>(input), input_length);
  return cudaGetLastError();
}

template <>
cudaError_t LaunchNegXPlus1Kernel<float>(cudaStream_t stream, int input_length, const float* input, float* output) {
  return _LaunchNegXPlus1Kernel(stream, input_length, input, output);
}

template <>
cudaError_t LaunchNegXPlus1Kernel<ortc::MFloat16>(cudaStream_t stream, int input_length, const ortc::MFloat16* input, ortc::MFloat16* output) {
  return _LaunchNegXPlus1Kernel(stream, input_length, input, output);
}
