// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_prop.cuh"
#include "utils.cuh"
#include "mul_sigmoid_impl.cuh"
#include "cuda_type.h"

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

using namespace Ort::Custom;

template <typename T> __device__ __inline__ T _exp_typed(const T x);

template <> __device__ __inline__ float _exp_typed(const float x) { return expf(x); }

#if __CUDA_ARCH__ < 700
template <> __device__ __inline__ half _exp_typed(const half x) {
  return __float2half(expf(__half2float(x)));
}
#else
template <> __device__ __inline__ half _exp_typed(const half x) { return hexp(x); }
#endif

template <typename T> __device__ __inline__ T sigmoid(const T a) {
  return a > T(0) ? (T)1 / ((T)1. + _exp_typed<T>(-a))
                  : (T)1 - (T)1 / ((T)1 + _exp_typed<T>(a));
}

#if __CUDA_ARCH__ < 700
template <> __device__ __inline__ half sigmoid(const half a) {
  return __float2half(sigmoid(__half2float(a)));
}
#endif

template <typename T> __device__ __inline__ T mul_sigmoid(const T a) { return a * sigmoid(a); }

#if __CUDA_ARCH__ < 700
template <> __device__ __inline__ half mul_sigmoid(const half a) {
  float x = __half2float(a);
  return __float2half(x * sigmoid(x));
}
#endif

template <typename T> __device__ __inline__ T mul_mul_sigmoid(const T x, const T y) {
  return x * y * sigmoid(y);
}

#if __CUDA_ARCH__ < 700
template <> __device__ __inline__ half mul_mul_sigmoid(const half x, const half y) {
  float hy = __half2float(y);
  return __float2half(__half2float(x) * hy * sigmoid(hy));
}
#endif

template <typename T>
__global__ void MulSigmoidKernel(T *output_data, const T *input_data, CUDA_LONG N) {
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= N)
    return;
  output_data[id] = mul_sigmoid(input_data[id]);
}

template <typename T>
__global__ void MulMulSigmoidKernel(T *output_data, const T *px, const T *py, CUDA_LONG N,
                                    CUDA_LONG Nx, CUDA_LONG Ny) {
  CUDA_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= N)
    return;
  output_data[id] = mul_mul_sigmoid(px[id % Nx], py[id % Ny]);
}

template <typename T>
cudaError_t _LaunchMulSigmoidKernel(cudaStream_t stream, int input_length, const T* input, T* output) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  using TT = typename contrib::CudaT<T>::MappedType;
  MulSigmoidKernel<TT><<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<TT*>(output), reinterpret_cast<const TT*>(input), input_length);
  return cudaGetLastError();
}

template <>
cudaError_t LaunchMulSigmoidKernel<float>(cudaStream_t stream, int input_length, const float* input, float* output) {
  return _LaunchMulSigmoidKernel(stream, input_length, input, output);
}

template <>
cudaError_t LaunchMulSigmoidKernel<ortc::MFloat16>(cudaStream_t stream, int input_length, const ortc::MFloat16* input, ortc::MFloat16* output) {
  return _LaunchMulSigmoidKernel(stream, input_length, input, output);
}

template <typename T>
cudaError_t _LaunchMulMulSigmoidKernel(cudaStream_t stream, int input_length_x, int input_length_y,
                                       const T* input_data_x, const T* input_data_y, T* output) {
  int input_length = std::max(input_length_x, input_length_y);
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  using TT = typename contrib::CudaT<T>::MappedType;
  MulMulSigmoidKernel<TT><<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<TT*>(output),
                                                              reinterpret_cast<const TT*>(input_data_x), 
                                                              reinterpret_cast<const TT*>(input_data_y),
                                                              input_length, input_length_x, input_length_y);
  return cudaGetLastError();
}

template <>
cudaError_t LaunchMulMulSigmoidKernel<float>(cudaStream_t stream, int input_length_x, int input_length_y,
                                             const float* input_data_x, const float* input_data_y, float* output) {
  return _LaunchMulMulSigmoidKernel(stream, input_length_x, input_length_y, input_data_x, input_data_y, output);
}

template <>
cudaError_t LaunchMulMulSigmoidKernel<ortc::MFloat16>(cudaStream_t stream, int input_length_x, int input_length_y,
                                                      const ortc::MFloat16* input_data_x, const ortc::MFloat16* input_data_y,
                                                      ortc::MFloat16* output) {
  return _LaunchMulMulSigmoidKernel(stream, input_length_x, input_length_y, input_data_x, input_data_y, output);
}
