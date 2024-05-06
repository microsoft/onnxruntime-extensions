// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_prop.cuh"
#include "utils.cuh"
#include "fast_gelu_impl.cuh"

using namespace Ort::Custom;

template <typename T>
__device__ __inline__ T _neg1plusx(const T x) {
  return (T)1 - x;
}

template <>
__device__ __inline__ half _neg1plusx(const half x) {
#if __CUDA_ARCH__ < 700
  return __float2half(1 - __half2float(x));
#else
  return (half)1 - x;
#endif
}

template <typename T>
__global__ void _NegXplus1Kernel(T* output_data, const T* input_data, int N) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= N)
    return;
  output_data[id] = _neg1plusx(input_data[id]);
}

template <typename T>
cudaError_t LaunchNegXPlus1Kernel(cudaStream_t stream, int input_length, const T* input, T* output) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  using TT = typename CudaT<T>::MappedType;
  NegXPlus1Kernel<TT, blockSize><<<gridSize, blockSize, 0, stream>>>((TT*)input, (TT*)output, input_length);
  return cudaGetLastError();
}
