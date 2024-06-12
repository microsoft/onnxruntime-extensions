// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_prop.cuh"
#include "utils.cuh"
#include "replace_zero_impl.cuh"
#include "cuda_type.h"

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

using namespace Ort::Custom;

template <typename T>
__device__ __inline__ T _replace_zero(const T x, const T by) {
  return x == (T)0 ? by : x;
}

template <>
__device__ __inline__ half _replace_zero(const half x, const half by) {
#if __CUDA_ARCH__ < 700
  return __half2float(x) == 0 ? by : x;
#else
  return x == (half)0 ? by : x;
#endif
}

template <typename T>
__global__ void ReplaceZeroKernel(T* output_data, const T* input_data, CUDA_LONG N, const T by) {
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= N)
    return;
  output_data[id] = _replace_zero(input_data[id], by);
}

template <typename T>
T _cast(float value) { return (T)value; }

template <>
half _cast(float value) { return __float2half(value); }

template <typename T>
cudaError_t _LaunchReplaceZeroKernel(cudaStream_t stream, int input_length, const T* input_data, T* output_data, float by) {
  if (input_length == 0)
    return cudaGetLastError();
  using TT = typename contrib::CudaT<T>::MappedType;

  CUDA_LONG N = static_cast<CUDA_LONG>(input_length);

  const int num_threads_per_block = 256;
  const int num_elements_per_thread = (N + num_threads_per_block - 1) / num_threads_per_block;

  TT cby = _cast<TT>(by);
  ReplaceZeroKernel<TT><<<num_elements_per_thread, num_threads_per_block, 0, stream>>>(
      reinterpret_cast<TT*>(output_data), reinterpret_cast<const TT*>(input_data), N, cby);
  return cudaGetLastError();
}

template <>
cudaError_t LaunchReplaceZeroKernel<float>(cudaStream_t stream, int input_length, const float* input_data, float* output_data, float by) {
  return _LaunchReplaceZeroKernel(stream, input_length, input_data, output_data, by);
}

template <>
cudaError_t LaunchReplaceZeroKernel<ortc::MFloat16>(cudaStream_t stream, int input_length, const ortc::MFloat16* input_data, ortc::MFloat16* output_data, float by) {
  return _LaunchReplaceZeroKernel(stream, input_length, input_data, output_data, by);
}
