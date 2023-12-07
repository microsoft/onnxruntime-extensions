// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_f16.h"
#include "fast_gelu_impl.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace Ort::Custom;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530) && ((__CUDACC_VER_MAJOR__ < 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ < 2)))
__device__ __forceinline__ half operator+(const half& lh, const half& rh) { return half((float)lh + (float)rh); }
__device__ __forceinline__ half operator-(const half& lh, const half& rh) { return half((float)lh - (float)rh); }
__device__ __forceinline__ half operator*(const half& lh, const half& rh) { return half((float)lh * (float)rh); }
__device__ __forceinline__ half operator/(const half& lh, const half& rh) { return half((float)lh / (float)rh); }

__device__ __forceinline__ half& operator+=(half& lh, const half& rh) {
  lh = half((float)lh + (float)rh);
  return lh;
}
__device__ __forceinline__ half& operator-=(half& lh, const half& rh) {
  lh = half((float)lh - (float)rh);
  return lh;
}
__device__ __forceinline__ half& operator*=(half& lh, const half& rh) {
  lh = half((float)lh * (float)rh);
  return lh;
}
__device__ __forceinline__ half& operator/=(half& lh, const half& rh) {
  lh = half((float)lh / (float)rh);
  return lh;
}

/* Note for increment and decrement we use the raw value 0x3C00 equating to half(1.0f), to avoid the extra conversion */
__device__ __forceinline__ __half& operator++(__half& h) {
  h = half((float)h + 1.0f);
  return h;
}
__device__ __forceinline__ __half& operator--(__half& h) {
  h = half((float)h - 1.0f);
  return h;
}
__device__ __forceinline__ __half operator++(__half& h, int) {
  half ret = h;
  h = half((float)h + 1);
  return ret;
}
__device__ __forceinline__ __half operator--(__half& h, int) {
  half ret = h;
  h = half((float)h - 1);
  return ret;
}

/* Unary plus and inverse operators */
__device__ __forceinline__ half operator+(const half& h) { return h; }
__device__ __forceinline__ half operator-(const half& h) { return half(-(float)h); }

/* Some basic comparison operations to make it look like a builtin */
__device__ __forceinline__ bool operator==(const half& lh, const half& rh) { return (float)lh == (float)rh; }
__device__ __forceinline__ bool operator!=(const half& lh, const half& rh) { return (float)lh != (float)rh; }
__device__ __forceinline__ bool operator>(const half& lh, const half& rh) { return (float)lh > (float)rh; }
__device__ __forceinline__ bool operator<(const half& lh, const half& rh) { return (float)lh < (float)rh; }
__device__ __forceinline__ bool operator>=(const half& lh, const half& rh) { return (float)lh >= (float)rh; }
__device__ __forceinline__ bool operator<=(const half& lh, const half& rh) { return (float)lh <= (float)rh; }

// support half2 arithmetic for cuda architecture < 5.3
__device__ __forceinline__ half2 operator+(const half2& lh, const half2& rh) {
  half2 r;
  r.x = lh.x + rh.x;
  r.y = lh.y + rh.y;
  return r;
}

__device__ __forceinline__ half2 operator-(const half2& lh, const half2& rh) {
  half2 r;
  r.x = lh.x - rh.x;
  r.y = lh.y - rh.y;
  return r;
}

__device__ __forceinline__ half2 operator*(const half2& lh, const half2& rh) {
  half2 r;
  r.x = lh.x * rh.x;
  r.y = lh.y * rh.y;
  return r;
}

__device__ __forceinline__ half2 operator/(const half2& lh, const half2& rh) {
  half2 r;
  r.x = lh.x / rh.x;
  r.y = lh.y / rh.y;
  return r;
}
#endif

/// Arithmetic for BFloat16

__device__ __forceinline__ BFloat16 operator+(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

__device__ __forceinline__ BFloat16 operator-(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

__device__ __forceinline__ BFloat16 operator*(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

__device__ __forceinline__ BFloat16 operator/(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

__device__ __forceinline__ BFloat16 operator-(const BFloat16& a) { return -static_cast<float>(a); }

__device__ __forceinline__ BFloat16& operator+=(BFloat16& a, const BFloat16& b) {
  a = a + b;
  return a;
}

__device__ __forceinline__ BFloat16& operator-=(BFloat16& a, const BFloat16& b) {
  a = a - b;
  return a;
}

__device__ __forceinline__ BFloat16& operator*=(BFloat16& a, const BFloat16& b) {
  a = a * b;
  return a;
}

__device__ __forceinline__ BFloat16& operator/=(BFloat16& a, const BFloat16& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

__device__ __forceinline__ float operator+(BFloat16 a, float b) { return a + b; }
__device__ __forceinline__ float operator-(BFloat16 a, float b) { return a - b; }
__device__ __forceinline__ float operator*(BFloat16 a, float b) { return a * b; }
__device__ __forceinline__ float operator/(BFloat16 a, float b) { return a / b; }

__device__ __forceinline__ float operator+(float a, BFloat16 b) { return a + b; }
__device__ __forceinline__ float operator-(float a, BFloat16 b) { return a - b; }
__device__ __forceinline__ float operator*(float a, BFloat16 b) { return a * b; }
__device__ __forceinline__ float operator/(float a, BFloat16 b) { return a / b; }

__device__ __forceinline__ float& operator+=(float& a, const BFloat16& b) { return a += b; }
__device__ __forceinline__ float& operator-=(float& a, const BFloat16& b) { return a -= b; }
__device__ __forceinline__ float& operator*=(float& a, const BFloat16& b) { return a *= b; }
__device__ __forceinline__ float& operator/=(float& a, const BFloat16& b) { return a /= b; }

/// Arithmetic with doubles

__device__ __forceinline__ double operator+(BFloat16 a, double b) { return static_cast<double>(a) + b; }
__device__ __forceinline__ double operator-(BFloat16 a, double b) { return static_cast<double>(a) - b; }
__device__ __forceinline__ double operator*(BFloat16 a, double b) { return static_cast<double>(a) * b; }
__device__ __forceinline__ double operator/(BFloat16 a, double b) { return static_cast<double>(a) / b; }

__device__ __forceinline__ double operator+(double a, BFloat16 b) { return a + static_cast<double>(b); }
__device__ __forceinline__ double operator-(double a, BFloat16 b) { return a - static_cast<double>(b); }
__device__ __forceinline__ double operator*(double a, BFloat16 b) { return a * static_cast<double>(b); }
__device__ __forceinline__ double operator/(double a, BFloat16 b) { return a / static_cast<double>(b); }

// Overloading < and > operators

__device__ __forceinline__ bool operator==(BFloat16& lhs, BFloat16& rhs) { return float(lhs) == float(rhs); }
__device__ __forceinline__ bool operator!=(BFloat16& lhs, BFloat16& rhs) { return float(lhs) != float(rhs); }
__device__ __forceinline__ bool operator>(BFloat16& lhs, BFloat16& rhs) { return float(lhs) > float(rhs); }
__device__ __forceinline__ bool operator<(BFloat16& lhs, BFloat16& rhs) { return float(lhs) < float(rhs); }

template <typename T>
__device__ __inline T _Tanh(T a);

template <>
__device__ __inline__ float _Tanh(float a) { return tanhf(a); }

template <>
__device__ __inline__ half _Tanh(half a) { return half(tanhf((float)a)); }

template <>
__device__ __inline__ half2 _Tanh(half2 a) {
  float2 tmp = (__half22float2(a));
  tmp.x = tanhf(tmp.x);
  tmp.y = tanhf(tmp.y);
  return __float22half2_rn(tmp);
}

template <>
__device__ __inline__ BFloat16 _Tanh(BFloat16 a) { return tanhf(static_cast<float>(a)); }

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

template <unsigned TPB>
__global__ void FastGeluKernel2(const half2 a, const half2 b, const half2 c, int input_length, int bias_length,
                                const half2* input, const half2* bias, half2* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < input_length) {
    const half2 x = input[idx];
    const half2 in = (bias == nullptr) ? x : (x + bias[idx % bias_length]);
    const half2 cdf = a + a * _Tanh(in * (c * in * in + b));
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

template <>
cudaError_t LaunchFastGeluKernel(cudaStream_t stream, int input_length, int bias_length,
                                 const half* input, const half* bias, half* output, bool use_half2) {
  constexpr int blockSize = 256;
  if (use_half2 && 0 == (bias_length & 1) /*&& prop.major >= 7*/ ) { // todo - get device id from ort for device property
    const int n = input_length / 2;
    const int gridSize = (n + blockSize - 1) / blockSize;
    const half2 A2 = __floats2half2_rn(A, A);
    const half2 B2 = __floats2half2_rn(B, B);
    const half2 C2 = __floats2half2_rn(C, C);
    const half2* input2 = reinterpret_cast<const half2*>(input);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    half2* output2 = reinterpret_cast<half2*>(output);
    FastGeluKernel2<blockSize><<<gridSize, blockSize, 0, stream>>>(A2, B2, C2, n, bias_length / 2,
                                                                   input2, bias2, output2);
  } else {
    const int gridSize = (input_length + blockSize - 1) / blockSize;
    FastGeluKernel<half, blockSize><<<gridSize, blockSize, 0, stream>>>(A, B, C, input_length, bias_length,
                                                                        input, bias, output);
  }

  return cudaGetLastError();
}

template <>
cudaError_t LaunchFastGeluKernel(cudaStream_t stream, int input_length, int bias_length,
                                 const BFloat16* input, const BFloat16* bias, BFloat16* output, bool /*use_half2*/) {
  constexpr int blockSize = 256;

  // remove nv_bfloat162 implementation for now to fix build issue
  // we can decide whether to add it back if there's perf concern
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  FastGeluKernel<BFloat16, blockSize>
      <<<gridSize, blockSize, 0, stream>>>(A, B, C, input_length, bias_length, input, bias, output);

  return cudaGetLastError();
}