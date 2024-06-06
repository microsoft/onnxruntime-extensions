// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_prop.cuh"
#include "utils.cuh"
#include "rotary_impl.cuh"
#include "cuda_type.h"

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

using namespace Ort::Custom;

template <typename T> __device__ __inline__ T _neg(const T x) { return -x; }

#if __CUDA_ARCH__ < 700
template <> __device__ __inline__ half _neg(const half x) {
  return __float2half(-__half2float(x));
}
#endif

template <typename T, RotarySide side>
__global__ void RotaryKernel(T *output_data, const T *input_data, CUDA_LONG half_N, CUDA_LONG half_stride) {
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= half_N)
    return;
  CUDA_LONG last = id % half_stride;
  id = (id - last) * 2 + last;
  if (side == RotarySide::RIGHT) {
    output_data[id + half_stride] = input_data[id];
    output_data[id] = _neg(input_data[id + half_stride]);
  } else {
    output_data[id + half_stride] = _neg(input_data[id]);
    output_data[id] = input_data[id + half_stride];
  }
}

template <typename T>
cudaError_t _LaunchRotaryKernel(cudaStream_t stream, int input_length, int last_dim,
                                const T* input_data, const int64_t* /* split_data */, T* output_data, RotarySide side) {
  if (input_length == 0)
      return cudaGetLastError();
  using TT = typename contrib::CudaT<T>::MappedType;

  CUDA_LONG N = static_cast<CUDA_LONG>(input_length);
  CUDA_LONG stride = static_cast<CUDA_LONG>(last_dim);

  const int num_threads_per_block = 256;
  const int num_elements_per_thread =
      (N / 2 + num_threads_per_block - 1) / num_threads_per_block;

  switch (side) {
  case RotarySide::LEFT:
    RotaryKernel<TT, RotarySide::LEFT>
        <<<num_elements_per_thread, num_threads_per_block, 0, stream>>>(reinterpret_cast<TT*>(output_data),
                                                                        reinterpret_cast<const TT*>(input_data),
                                                                        N / 2, stride / 2);
    break;
  case RotarySide::RIGHT:
    RotaryKernel<TT, RotarySide::RIGHT>
        <<<num_elements_per_thread, num_threads_per_block, 0, stream>>>(reinterpret_cast<TT*>(output_data),
                                                                        reinterpret_cast<const TT*>(input_data),
                                                                        N / 2, stride / 2);
    break;
  }
  return cudaGetLastError();
}

template <>
cudaError_t LaunchRotaryKernel<float>(cudaStream_t stream, int input_length, int last_dim,
                                      const float* input_data, const int64_t* split_data, float* output_data, RotarySide side) {
  return _LaunchRotaryKernel(stream, input_length, last_dim, input_data, split_data, output_data, side);
}

template <>
cudaError_t LaunchRotaryKernel<ortc::MFloat16>(cudaStream_t stream, int input_length, int last_dim,
                                               const ortc::MFloat16* input_data, const int64_t* split_data,
                                               ortc::MFloat16* output_data, RotarySide side) {
  return _LaunchRotaryKernel(stream, input_length, last_dim, input_data, split_data, output_data, side);
}
