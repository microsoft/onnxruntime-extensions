// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_prop.cuh"
#include "utils.cuh"
#include "add_mul_impl.cuh"
#include "cuda_type.h"

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

using namespace Ort::Custom;

__device__ __forceinline__ void _add3_op(float* ab, float* ac, const float a, const float b, const float c) {
  *ab = a + b;
  *ac = a + c;
}

__device__ __forceinline__ void _add3_op(half* ab, half* ac, const half a, const half b, const half c) {
#if __CUDA_ARCH__ < 700
  *ab = __float2half(__half2float(a) + __half2float(b));
  *ac = __float2half(__half2float(a) + __half2float(c));
#else
  *ab = a + b;
  *ac = a + c;
#endif
}

__device__ __forceinline__ void _mul3_op(float* ab, float* ac, const float a, const float b, const float c) {
  *ab = a * b;
  *ac = a * c;
}

__device__ __forceinline__ void _mul3_op(half* ab, half* ac, const half a, const half b, const half c) {
#if __CUDA_ARCH__ < 700
  *ab = __float2half(__half2float(a) * __half2float(b));
  *ac = __float2half(__half2float(a) * __half2float(c));
#else
  *ab = a * b;
  *ac = a * c;
#endif
}

template <typename T>
struct Mul3SharedOp {
  __device__ __forceinline__ void operator()(T* ab, T* ac, const T a, const T b, const T c) const {
    _mul3_op(ab, ac, a, b, c);
  }
};

template <typename T>
struct Add3SharedOp {
  __device__ __forceinline__ void operator()(T* ab, T* ac, const T a, const T b, const T c) const {
    _add3_op(ab, ac, a, b, c);
  }
};

template <typename T, typename TFunc, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void AddMulKernel(T* output_ab, T* output_ac, const T* pA, const T* pB,
                             const T* pC, CUDA_LONG nA, CUDA_LONG nB, CUDA_LONG nC,
                             CUDA_LONG N, const TFunc func) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      func(output_ab + id, output_ac + id, pA[id % nA], pB[id % nB], pC[id % nC]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
cudaError_t _LaunchAddOrMulSharedInputKernel(cudaStream_t stream,
                                             const T* pA, const T* pB, const T* pC,
                                             T* output_ab, T* output_ac,
                                             int64_t countA, int64_t countB, int64_t countC, bool addition) {
  int64_t max_count = std::max(std::max(countA, countB), countC);
  if (max_count == 0)  // special case where there's a dim value of 0 in the output shape
    return cudaGetLastError();

  const int num_elements_per_thread = 4;
  const int num_threads_per_block = 256;
  const int num_el_th = num_threads_per_block * num_elements_per_thread;

  int blocksPerGrid = (max_count + num_el_th - 1) / num_el_th;

  using TT = typename contrib::CudaT<T>::MappedType;

  if (addition) {
    AddMulKernel<TT, Add3SharedOp<TT>, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            reinterpret_cast<TT*>(output_ab), reinterpret_cast<TT*>(output_ac),
            reinterpret_cast<const TT*>(pA), reinterpret_cast<const TT*>(pB), reinterpret_cast<const TT*>(pC), static_cast<CUDA_LONG>(countA),
            static_cast<CUDA_LONG>(countB), static_cast<CUDA_LONG>(countC),
            static_cast<CUDA_LONG>(max_count), Add3SharedOp<TT>());
  } else {
    AddMulKernel<TT, Mul3SharedOp<TT>, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            reinterpret_cast<TT*>(output_ab), reinterpret_cast<TT*>(output_ac),
            reinterpret_cast<const TT*>(pA), reinterpret_cast<const TT*>(pB), reinterpret_cast<const TT*>(pC), static_cast<CUDA_LONG>(countA),
            static_cast<CUDA_LONG>(countB), static_cast<CUDA_LONG>(countC),
            static_cast<CUDA_LONG>(max_count), Mul3SharedOp<TT>());
  }
  return cudaGetLastError();
}

template <>
cudaError_t LaunchAddOrMulSharedInputKernel<float>(cudaStream_t stream, const float* input_a, const float* input_b, const float* input_c,
                                                   float* output_ab, float* output_ac,
                                                   int64_t length_a, int64_t length_b, int64_t length_c, bool addition) {
  return _LaunchAddOrMulSharedInputKernel(stream, input_a, input_b, input_c, output_ab, output_ac, length_a, length_b, length_c, addition);
}

template <>
cudaError_t LaunchAddOrMulSharedInputKernel<ortc::MFloat16>(cudaStream_t stream, const ortc::MFloat16* input_a, const ortc::MFloat16* input_b, const ortc::MFloat16* input_c,
                                                            ortc::MFloat16* output_ab, ortc::MFloat16* output_ac,
                                                            int64_t length_a, int64_t length_b, int64_t length_c, bool addition) {
  return _LaunchAddOrMulSharedInputKernel(stream, input_a, input_b, input_c, output_ab, output_ac, length_a, length_b, length_c, addition);
}
