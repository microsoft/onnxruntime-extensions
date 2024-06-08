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

__device__ __forceinline__ void _add3_2_op(float* ab, float* ac, const float a, const float b, const float c) {
  *ab = a + b;
  *ac = a + c;
}

__device__ __forceinline__ void _add3_2_op(half* ab, half* ac, const half a, const half b, const half c) {
#if __CUDA_ARCH__ < 700
  *ab = __float2half(__half2float(a) + __half2float(b));
  *ac = __float2half(__half2float(a) + __half2float(c));
#else
  *ab = a + b;
  *ac = a + c;
#endif
}

__device__ __forceinline__ void _mul3_2_op(float* ab, float* ac, const float a, const float b, const float c) {
  *ab = a * b;
  *ac = a * c;
}

__device__ __forceinline__ void _mul3_2_op(half* ab, half* ac, const half a, const half b, const half c) {
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
    _mul3_2_op(ab, ac, a, b, c);
  }
};

template <typename T>
struct Add3SharedOp {
  __device__ __forceinline__ void operator()(T* ab, T* ac, const T a, const T b, const T c) const {
    _add3_2_op(ab, ac, a, b, c);
  }
};

template <typename T, typename TFunc, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void AddMulSharedInputKernel(T* output_ab, T* output_ac, const T* pA, const T* pB,
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
    AddMulSharedInputKernel<TT, Add3SharedOp<TT>, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            reinterpret_cast<TT*>(output_ab), reinterpret_cast<TT*>(output_ac),
            reinterpret_cast<const TT*>(pA), reinterpret_cast<const TT*>(pB), reinterpret_cast<const TT*>(pC), static_cast<CUDA_LONG>(countA),
            static_cast<CUDA_LONG>(countB), static_cast<CUDA_LONG>(countC),
            static_cast<CUDA_LONG>(max_count), Add3SharedOp<TT>());
  } else {
    AddMulSharedInputKernel<TT, Mul3SharedOp<TT>, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            reinterpret_cast<TT*>(output_ab), reinterpret_cast<TT*>(output_ac),
            reinterpret_cast<const TT*>(pA), reinterpret_cast<const TT*>(pB), reinterpret_cast<const TT*>(pC), static_cast<CUDA_LONG>(countA),
            static_cast<CUDA_LONG>(countB), static_cast<CUDA_LONG>(countC),
            static_cast<CUDA_LONG>(max_count), Mul3SharedOp<TT>());
  }
  return cudaGetLastError();
}

template <>
cudaError_t LaunchAddOrMulSharedInputKernel<float>(cudaStream_t stream,
                                                   const float* input_a, const float* input_b, const float* input_c,
                                                   float* output_ab, float* output_ac,
                                                   int64_t length_a, int64_t length_b, int64_t length_c, bool addition) {
  return _LaunchAddOrMulSharedInputKernel(stream, input_a, input_b, input_c,
                                          output_ab, output_ac,
                                          length_a, length_b, length_c, addition);
}

template <>
cudaError_t LaunchAddOrMulSharedInputKernel<ortc::MFloat16>(cudaStream_t stream,
                                                            const ortc::MFloat16* input_a, const ortc::MFloat16* input_b, const ortc::MFloat16* input_c,
                                                            ortc::MFloat16* output_ab, ortc::MFloat16* output_ac,
                                                            int64_t length_a, int64_t length_b, int64_t length_c, bool addition) {
  return _LaunchAddOrMulSharedInputKernel(stream, input_a, input_b, input_c,
                                          output_ab, output_ac,
                                          length_a, length_b, length_c, addition);
}

__device__ __forceinline__ void _add3_op(float* address, const float a, const float b,
                                         const float c) {
  *address = a + b + c;
}

__device__ __forceinline__ void _add3_op(half* address, const half a, const half b,
                                         const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half(__half2float(a) + __half2float(b) + __half2float(c));
#else
  *address = a + b + c;
#endif
}

__device__ __forceinline__ void _mul3_op(float* address, const float a, const float b,
                                         const float c) {
  *address = a * b * c;
}

__device__ __forceinline__ void _mul3_op(half* address, const half a, const half b,
                                         const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half(__half2float(a) * __half2float(b) * __half2float(c));
#else
  *address = a * b * c;
#endif
}

template <typename T>
struct Mul3Op {
  __device__ __inline__ void operator()(T* address, const T a, const T b, const T c) const {
    _mul3_op(address, a, b, c);
  }
};

template <typename T>
struct Add3Op {
  __device__ __inline__ void operator()(T* address, const T a, const T b, const T c) const {
    _add3_op(address, a, b, c);
  }
};

template <typename T, typename TFunc, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void AddMulTwiceKernel(T* output, const T* pA, const T* pB,
                                  const T* pC, CUDA_LONG nA, CUDA_LONG nB, CUDA_LONG nC,
                                  CUDA_LONG N, const TFunc func) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      func(output_ab, pA[id % nA], pB[id % nB], pC[id % nC]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
cudaError_t _LaunchAddOrMulTwiceKernel(cudaStream_t stream,
                                       const T* pA, const T* pB, const T* pC,
                                       T* output,
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
    AddMulTwiceKernel<TT, Add3Op<TT>, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            reinterpret_cast<TT*>(output),
            reinterpret_cast<const TT*>(pA), reinterpret_cast<const TT*>(pB), reinterpret_cast<const TT*>(pC),
            static_cast<CUDA_LONG>(countA), static_cast<CUDA_LONG>(countB), static_cast<CUDA_LONG>(countC),
            static_cast<CUDA_LONG>(max_count), Add3SharedOp<TT>());
  } else {
    AddMulTwiceKernel<TT, Mul3Op<TT>, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            reinterpret_cast<TT*>(output),
            reinterpret_cast<const TT*>(pA), reinterpret_cast<const TT*>(pB), reinterpret_cast<const TT*>(pC), static_cast<CUDA_LONG>(countA),
            static_cast<CUDA_LONG>(countB), static_cast<CUDA_LONG>(countC),
            static_cast<CUDA_LONG>(max_count), Mul3SharedOp<TT>());
  }
  return cudaGetLastError();
}

template <>
cudaError_t LaunchAddOrMulSharedInputKernel<float>(cudaStream_t stream,
                                                   const float* input_a, const float* input_b, const float* input_c,
                                                   float* output,
                                                   int64_t length_a, int64_t length_b, int64_t length_c, bool addition) {
  return _LaunchAddOrMulSharedInputKernel(stream, input_a, input_b, input_c,
                                          output,
                                          length_a, length_b, length_c, addition);
}

template <>
cudaError_t LaunchAddOrMulSharedInputKernel<ortc::MFloat16>(cudaStream_t stream,
                                                            const ortc::MFloat16* input_a, const ortc::MFloat16* input_b, const ortc::MFloat16* input_c,
                                                            ortc::MFloat16* output,
                                                            int64_t length_a, int64_t length_b, int64_t length_c, bool addition) {
  return _LaunchAddOrMulSharedInputKernel(stream, input_a, input_b, input_c,
                                          output,
                                          length_a, length_b, length_c, addition);
}

__device__ __forceinline__ void _addmul_op(float* address, const float a, const float b,
                                           const float c) {
  *address = (a + b) * c;
}

__device__ __forceinline__ void _addmul_op(half* address, const half a, const half b,
                                           const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half((__half2float(a) + __half2float(b)) * __half2float(c));
#else
  *address = (a + b) * c;
#endif
}

__device__ __forceinline__ void _muladd_op(float* address, const float a, const float b,
                                           const float c) {
  *address = a * b + c;
}

__device__ __forceinline__ void _muladd_op(half* address, const half a, const half b,
                                           const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half(__half2float(a) * __half2float(b) + __half2float(c));
#else
  *address = a * b + c;
#endif
}

template <typename T>
struct AddMul {
  __device__ __inline__ void operator()(T* address, const T a, const T b, const T c) const {
    _addmul_op(address, a, b, c);
  }
};

template <typename T>
struct MulAdd {
  __device__ __inline__ void operator()(T* address, const T a, const T b, const T c) const {
    _muladd_op(address, a, b, c);
  }
};

template <typename T, typename TFunc, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _AddAndMulKernel(T* output_data, const T* pA, const T* pB, const T* pC,
                                 CUDA_LONG nA, CUDA_LONG nB, CUDA_LONG nC, CUDA_LONG N,
                                 const TFunc func) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      func(output_data + id, pA[id % nA], pB[id % nB], pC[id % nC]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, typename TFunc, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _AddAndMulSwitchMiddleAxesKernel(T* output_data, const T* pA, const T* pB,
                                                 const T* pC, CUDA_LONG nA, CUDA_LONG nB,
                                                 CUDA_LONG nC, CUDA_LONG N,
                                                 const TFunc func, CUDA_LONG d2,
                                                 CUDA_LONG d3, CUDA_LONG d4) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  CUDA_LONG id = start;
  CUDA_LONG k, j, ido;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      k = (id / d4) % d3;
      j = (id / (d4 * d3)) % d2;
      ido = id + d4 * ((k * d2 + j) - (j * d3 + k));
      func(output_data + ido, pA[id % nA], pB[id % nB], pC[id % nC]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
cudaError_t _LaunchAddAndMulKernel(cudaStream_t stream,
                                   const T* pA, const T* pB, const T* pC,
                                   T* output,
                                   int64_t countA, int64_t countB, int64_t countC,
                                   bool addition_first) {
  int64_t max_count = std::max(std::max(countA, countB), countC);
  if (max_count == 0)  // special case where there's a dim value of 0 in the output shape
    return cudaGetLastError();

  const int num_elements_per_thread = 4;
  const int num_threads_per_block = 256;
  const int num_el_th = num_threads_per_block * num_elements_per_thread;

  int blocksPerGrid = (max_count + num_el_th - 1) / num_el_th;

  using TT = typename contrib::CudaT<T>::MappedType;

  if (addition_first) {
    AddAndMulKernel<TT, AddMul<TT>, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            cuda_stream,
            reinterpret_cast<TT*>(output),
            reinterpret_cast<const TT*>(pA),
            reinterpret_cast<const TT*>(pB),
            reinterpret_cast<const TT*>(pC),
            countA, countB, countC,
            max_size, AddMul<TT>());
  } else {
    AddAndMulKernel<TT, MulAdd<TT>, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            cuda_stream,
            reinterpret_cast<TT*>(output),
            reinterpret_cast<const TT*>(pA),
            reinterpret_cast<const TT*>(pB),
            reinterpret_cast<const TT*>(pC),
            countA, countB, countC,
            max_size, MulAdd<TT>());
  }
  return cudaGetLastError();
}

template <>
cudaError_t LaunchAddAndMulKernel(cudaStream_t stream, const float* input_a, const float* input_b, const float* input_c,
                                  float* output, int64_t length_a, int64_t length_b, int64_t length_c,
                                  bool addition) {
  return _LaunchAddAndMulKernel(stream, pA, pB, pC, output, countA, countB, countC, addition_first);
}

template <>
cudaError_t LaunchAddAndMulKernel(cudaStream_t stream,
                                  const ortc::MFloat16* input_a, const ortc::MFloat16* input_b,
                                  const ortc::MFloat16* input_c,
                                  ortc::MFloat16* output, int64_t length_a, int64_t length_b, int64_t length_c,
                                  bool addition) {
  return _LaunchAddAndMulKernel(stream, pA, pB, pC, output, countA, countB, countC, addition_first);
}

template <typename T>
cudaError_t _LaunchAddAndMulSwitchMiddleAxesKernel(cudaStream_t stream,
                                                   const T* pA, const T* pB, const T* pC,
                                                   T* output,
                                                   int64_t countA, int64_t countB, int64_t countC,
                                                   bool addition_first, int64_t d2, int64_t d3, int64_t d4) {
  int64_t max_count = std::max(std::max(countA, countB), countC);
  if (max_count == 0)  // special case where there's a dim value of 0 in the output shape
    return cudaGetLastError();

  const int num_elements_per_thread = 4;
  const int num_threads_per_block = 256;
  const int num_el_th = num_threads_per_block * num_elements_per_thread;

  int blocksPerGrid = (max_count + num_el_th - 1) / num_el_th;

  using TT = typename contrib::CudaT<T>::MappedType;

  if (addition_first) {
    AddAndMulSwitchMiddleAxesKernel<TT, AddMul<TT>, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            cuda_stream,
            reinterpret_cast<TT*>(output),
            reinterpret_cast<const TT*>(pA),
            reinterpret_cast<const TT*>(pB),
            reinterpret_cast<const TT*>(pC),
            countA, countB, countC,
            max_size, AddMul<TT>());
  } else {
    AddAndMulSwitchMiddleAxesKernel<TT, MulAdd<TT>, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            cuda_stream,
            reinterpret_cast<TT*>(output),
            reinterpret_cast<const TT*>(pA),
            reinterpret_cast<const TT*>(pB),
            reinterpret_cast<const TT*>(pC),
            countA, countB, countC,
            max_size, MulAdd<TT>());
  }
  return cudaGetLastError();
}

template <>
cudaError_t LaunchAddAndMulSwitchMiddleAxesKernel(cudaStream_t stream, const float* input_a, const float* input_b, const float* input_c,
                                                  float* output, int64_t length_a, int64_t length_b, int64_t length_c,
                                                  bool addition,
                                                  int64_t d2, int64_t d3, int64_t d4) {
  return _LaunchAddAndMulSwitchMiddleAxesKernel(stream, pA, pB, pC, output, countA, countB, countC,
                                                addition_first, d2, d3, d4);
}

template <>
cudaError_t LaunchAddAndMulSwitchMiddleAxesKernel(cudaStream_t stream, const ortc::MFloat16* input_a,
                                                  const ortc::MFloat16* input_b, const ortc::MFloat16* input_c,
                                                  ortc::MFloat16* output, int64_t length_a, int64_t length_b, int64_t length_c,
                                                  bool addition,
                                                  int64_t d2, int64_t d3, int64_t d4) {
  return _LaunchAddAndMulSwitchMiddleAxesKernel(stream, pA, pB, pC, output, countA, countB, countC,
                                                addition_first, d2, d3, d4);
}

__device__ __forceinline__ void _submul_op(float* address, const float a, const float b,
                                           const float c) {
  *address = (a - b) * c;
}

__device__ __forceinline__ void _submul_op(half* address, const half a, const half b,
                                           const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half((__half2float(a) - __half2float(b)) * __half2float(c));
#else
  *address = (a - b) * c;
#endif
}

__device__ __forceinline__ void _submul_neg_op(float* address, const float a, const float b,
                                               const float c) {
  *address = (b - a) * c;
}

__device__ __forceinline__ void _submul_neg_op(half* address, const half a, const half b,
                                               const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half((__half2float(b) - __half2float(a)) * __half2float(c));
#else
  *address = (b - a) * c;
#endif
}

__device__ __forceinline__ void _mulsub_op(float* address, const float a, const float b,
                                           const float c) {
  *address = a * b - c;
}

__device__ __forceinline__ void _mulsub_op(half* address, const half a, const half b,
                                           const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half(__half2float(a) * __half2float(b) - __half2float(c));
#else
  *address = a * b - c;
#endif
}

__device__ __forceinline__ void _mulsub_neg_op(float* address, const float a, const float b,
                                               const float c) {
  *address = c - a * b;
}

__device__ __forceinline__ void _mulsub_neg_op(half* address, const half a, const half b,
                                               const half c) {
#if __CUDA_ARCH__ < 700
  *address = __float2half(__half2float(c) - __half2float(a) * __half2float(b));
#else
  *address = c - a * b;
#endif
}

template <typename T>
struct SubMul {
  __device__ __inline__ void operator()(T* address, const T a, const T b, const T c) const {
    _submul_op(address, a, b, c);
  }
};

template <typename T>
struct MulSub {
  __device__ __inline__ void operator()(T* address, const T a, const T b, const T c) const {
    _mulsub_op(address, a, b, c);
  }
};

template <typename T>
struct SubMulNeg {
  __device__ __inline__ void operator()(T* address, const T a, const T b, const T c) const {
    _submul_neg_op(address, a, b, c);
  }
};

template <typename T>
struct MulSubNeg {
  __device__ __inline__ void operator()(T* address, const T a, const T b, const T c) const {
    _mulsub_neg_op(address, a, b, c);
  }
};

template <typename T, typename TFunc, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _MulSubKernel(T* output_data, const T* pA, const T* pB, const T* pC,
                              CUDA_LONG nA, CUDA_LONG nB, CUDA_LONG nC, CUDA_LONG N,
                              const TFunc func) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      func(output_data + id, pA[id % nA], pB[id % nB], pC[id % nC]);
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
cudaError_t _LaunchSubAndMulKernel(cudaStream_t stream,
                                   const T* pA, const T* pB, const T* pC,
                                   T* output,
                                   int64_t countA, int64_t countB, int64_t countC,
                                   bool addition_first) {
  int64_t max_count = std::max(std::max(countA, countB), countC);
  if (max_count == 0)  // special case where there's a dim value of 0 in the output shape
    return cudaGetLastError();

  const int num_elements_per_thread = 4;
  const int num_threads_per_block = 256;
  const int num_el_th = num_threads_per_block * num_elements_per_thread;

  int blocksPerGrid = (max_count + num_el_th - 1) / num_el_th;

  using TT = typename contrib::CudaT<T>::MappedType;

  if (addition_first) {
    if (negative) {
      SubAndMulKernel<TT, SubMul<TT>, num_threads_per_block, num_elements_per_thread>
          <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
              cuda_stream,
              reinterpret_cast<TT*>(output),
              reinterpret_cast<const TT*>(pA),
              reinterpret_cast<const TT*>(pB),
              reinterpret_cast<const TT*>(pC),
              countA, countB, countC,
              max_size, SubMulNEg<TT>());
    } else {
      SubAndMulKernel<TT, SubMul<TT>, num_threads_per_block, num_elements_per_thread>
          <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
              cuda_stream,
              reinterpret_cast<TT*>(output),
              reinterpret_cast<const TT*>(pA),
              reinterpret_cast<const TT*>(pB),
              reinterpret_cast<const TT*>(pC),
              countA, countB, countC,
              max_size, SubMul<TT>());
    }
  } else {
    if (negative) {
      SubAndMulKernel<TT, MulSub<TT>, num_threads_per_block, num_elements_per_thread>
          <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
              cuda_stream,
              reinterpret_cast<TT*>(output),
              reinterpret_cast<const TT*>(pA),
              reinterpret_cast<const TT*>(pB),
              reinterpret_cast<const TT*>(pC),
              countA, countB, countC,
              max_size, MulSubNeg<TT>());
    } else {
      SubAndMulKernel<TT, MulSub<TT>, num_threads_per_block, num_elements_per_thread>
          <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
              cuda_stream,
              reinterpret_cast<TT*>(output),
              reinterpret_cast<const TT*>(pA),
              reinterpret_cast<const TT*>(pB),
              reinterpret_cast<const TT*>(pC),
              countA, countB, countC,
              max_size, MulSub<TT>());
    }
  }
  return cudaGetLastError();
}

template <>
cudaError_t LaunchSubAndMulKernel(cudaStream_t stream, const float* input_a, const float* input_b, const float* input_c,
                                  float* output, int64_t length_a, int64_t length_b, int64_t length_c,
                                  bool subtract_first, bool negative) {
  return _LaunchSubAndMulKernel(stream, pA, pB, pC, output, countA, countB, countC, subtract_first, negative);
}

template <>
cudaError_t LaunchSubAndMulKernel(cudaStream_t stream,
                                  const ortc::MFloat16* input_a, const ortc::MFloat16* input_b,
                                  const ortc::MFloat16* input_c,
                                  ortc::MFloat16* output, int64_t length_a, int64_t length_b, int64_t length_c,
                                  bool subtract_first, negative) {
  return _LaunchSubAndMulKernel(stream, pA, pB, pC, output, countA, countB, countC, subtract_first, negative);
}
