// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_prop.cuh"
#include "utils.cuh"
#include "scatter_nd_of_shape_impl.cuh"
#include "cuda_type.h"

namespace contrib {

#define _ENFORCE(cond, msg) \
  if (!(cond)) ORTX_CXX_API_THROW(msg, ORT_RUNTIME_EXCEPTION);

#ifndef HIP_LONG
#define HIP_LONG int32_t
#endif

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

template <typename T>
__device__ __forceinline__ void _add_inplace(T& x, const T a) { x += a; }

template <>
__device__ __forceinline__ void _add_inplace(half& x, const half a) {
#if __CUDA_ARCH__ < 700
  x = __float2half(__half2float(x) + __half2float(a));
#else
  x += a;
#endif
}

template <typename T>
__global__ void
addition_inplace_kernel(T* __restrict__ output_data, const int64_t* __restrict__ indices_data,
                        const T* __restrict__ updates_data, const CUDA_LONG indice_size,
                        const CUDA_LONG nrows, const CUDA_LONG stride) {
  HIP_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= stride)
    return;

  for (size_t i = 0; i < nrows; ++i) {
    output_data[i * stride + id] = 0;
  }

  int64_t index;
  for (size_t i = 0; i < indice_size; ++i) {
    index = (indices_data[i] + nrows) % nrows;
    _add_inplace(output_data[index * stride + id], updates_data[i * stride + id]);
  }
}

template <typename T>
__global__ void masked_addition_inplace_kernel(T *__restrict__ output_data,
                                               const int64_t *__restrict__ indices_data,
                                               const T *__restrict__ updates_data,
                                               const CUDA_LONG indice_size,
                                               const CUDA_LONG nrows, const CUDA_LONG stride,
                                               const int64_t masked_value) {
  auto id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= stride)
    return;

  for (size_t i = 0; i < nrows; ++i) {
    output_data[i * stride + id] = 0;
  }

  for (size_t i = 0; i < indice_size; ++i) {
    if (indices_data[i] == masked_value)
      continue;
    _add_inplace(output_data[indices_data[i] * stride + id], updates_data[i * stride + id]);
  }
}

template <typename T, int NTHREAD>
__global__ void masked_addition_inplace_kernelN(T *__restrict__ output_data,
                                                const int64_t *__restrict__ indices_data,
                                                const T *__restrict__ updates_data,
                                                const CUDA_LONG indice_size,
                                                const CUDA_LONG nrows, const CUDA_LONG stride,
                                                const int64_t masked_value) {
  __shared__ int64_t shared_indices[NTHREAD];

  CUDA_LONG tid = threadIdx.x;
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;

  for (size_t i = 0; i < nrows; ++i) {
    output_data[i * stride + id] = 0;
  }

  int begin = 0;
  int end = std::min(begin + NTHREAD, indice_size);
  while (begin < end && (end == begin + NTHREAD)) {
    shared_indices[tid] = indices_data[tid + begin];
    __syncthreads();

    for (size_t i = begin; i < end; ++i) {
      if (shared_indices[tid] == masked_value)
        continue;
      _add_inplace(output_data[shared_indices[tid] * stride + id],
                   updates_data[i * stride + id]);
    }

    begin = end;
    end = std::min(begin + NTHREAD, indice_size);
  }

  for (size_t i = begin; i < indice_size; ++i) {
    if (indices_data[i] == masked_value)
      continue;
    _add_inplace(output_data[indices_data[i] * stride + id], updates_data[i * stride + id]);
  }
}

template <class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values, size_t first = 0) {
  NTYPE r = 1;
  for (auto it = values.begin() + first; it != values.end(); ++it)
    r *= *it;
  return r;
}

template <typename T>
cudaError_t ScatterNDOfShapeKernel(cudaStream_t stream,
                                   const std::vector<int64_t>& output_shape,
                                   const std::vector<int64_t>& indices_shape,
                                   const int64_t* indices_data,
                                   const T* updates_data,
                                   T* output_data,
                                   ScatterReduction reduction) {
  if (reduction != ScatterReduction::Add)
    ORTX_CXX_API_THROW("Only reduction 'add' is implemented.", ORT_RUNTIME_EXCEPTION);
  size_t indice_size = static_cast<size_t>(flattened_dimension(indices_shape));
  size_t output_size = static_cast<size_t>(flattened_dimension(output_shape));
  size_t rank = output_shape.size() - indices_shape.size();
  size_t stride =  static_cast<size_t>(flattened_dimension(output_shape, output_shape.size() - 1 - rank));
  size_t nrows = output_size / stride;

  int threads_per_block = 256;
  int blocks_per_grid = (stride + threads_per_block - 1) / threads_per_block;

  dim3 threads(threads_per_block);
  dim3 blocks(blocks_per_grid);
  using TT = typename CudaT<T>::MappedType;
  addition_inplace_kernel<TT><<<blocks, threads, 0, stream>>>(reinterpret_cast<TT*>(output_data), indices_data,
                                                              reinterpret_cast<const TT*>(updates_data),
                                                              indice_size, nrows, stride);
  return cudaGetLastError();
}

template <typename T>
cudaError_t MaskedScatterNDOfShapeKernel(cudaStream_t stream, const std::vector<int64_t> &input_shape,
                                         const std::vector<int64_t> &indices_shape,
                                         const int64_t *indices_data, const T *updates_data,
                                         T *output_data,
                                         ScatterReduction reduction, int64_t masked_value) {
  if (reduction != ScatterReduction::Add)
    ORTX_CXX_API_THROW("Only reduction 'add' is implemented.", ORT_RUNTIME_EXCEPTION);
  size_t indice_size = static_cast<size_t>(flattened_dimension(indices_shape));
  size_t input_size = static_cast<size_t>(flattened_dimension(input_shape));
  size_t stride = input_shape[input_shape.size() - 1];
  size_t nrows = input_size / stride;

  std::vector<size_t> next_batch(indice_size);
  std::vector<uint8_t> processed(input_shape[0], 0);
  std::vector<uint8_t> processed_once(input_shape[0], 0);

  int threads_per_block = 256;
  bool split = stride / threads_per_block <= 32;

  int blocks_per_grid = (stride + threads_per_block - 1) / threads_per_block;
  dim3 threads(threads_per_block);
  dim3 blocks(blocks_per_grid);

  using TT = typename CudaT<T>::MappedType;

  if (split && stride >= 256 && threads_per_block == 256) {
    masked_addition_inplace_kernelN<TT, 256><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<TT*>(output_data), indices_data,
        reinterpret_cast<const TT*>(updates_data),
        indice_size, nrows, stride, masked_value);
  } else {
    masked_addition_inplace_kernel<TT><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<TT*>(output_data), indices_data,
        reinterpret_cast<const TT*>(updates_data),
        indice_size, nrows, stride, masked_value);
  }
  return cudaGetLastError();
}

template <>
cudaError_t LaunchScatterNDOfShapeKernel<float>(cudaStream_t stream,
                                                const std::vector<int64_t>& output_shape,
                                                const std::vector<int64_t>& indices_shape,
                                                const int64_t* indices,
                                                const float* updates,
                                                float* output,
                                                ScatterReduction reduction) {
  return ScatterNDOfShapeKernel(stream,
                                output_shape,
                                indices_shape,
                                indices,
                                updates,
                                output,
                                reduction);
}

template <>
cudaError_t LaunchScatterNDOfShapeKernel<ortc::MFloat16>(cudaStream_t stream,
                                                         const std::vector<int64_t>& output_shape,
                                                         const std::vector<int64_t>& indices_shape,
                                                         const int64_t* indices,
                                                         const ortc::MFloat16* updates,
                                                         ortc::MFloat16* output,
                                                         ScatterReduction reduction) {
  return ScatterNDOfShapeKernel(stream,
                                output_shape,
                                indices_shape,
                                indices,
                                updates,
                                output,
                                reduction);
}

template <>
cudaError_t LaunchMaskedScatterNDOfShapeKernel<float>(cudaStream_t stream,
                                                      const std::vector<int64_t>& output_shape,
                                                      const std::vector<int64_t>& indices_shape,
                                                      const int64_t* indices,
                                                      const float* updates,
                                                      float* output,
                                                      ScatterReduction reduction,
                                                      int64_t masked_value) {
  return MaskedScatterNDOfShapeKernel(stream,
                                      output_shape,
                                      indices_shape,
                                      indices,
                                      updates,
                                      output,
                                      reduction,
                                      masked_value);
}

template <>
cudaError_t LaunchMaskedScatterNDOfShapeKernel<ortc::MFloat16>(cudaStream_t stream,
                                                               const std::vector<int64_t>& output_shape,
                                                               const std::vector<int64_t>& indices_shape,
                                                               const int64_t* indices,
                                                               const ortc::MFloat16* updates,
                                                               ortc::MFloat16* output,
                                                               ScatterReduction reduction,
                                                               int64_t masked_value) {
  return MaskedScatterNDOfShapeKernel(stream,
                                      output_shape,
                                      indices_shape,
                                      indices,
                                      updates,
                                      output,
                                      reduction,
                                      masked_value);
}

}  // namespace contrib
