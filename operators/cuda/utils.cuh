// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_f16.h"
#include "string_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

namespace ortc = Ort::Custom;

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

/// Arithmetic for ortc::BFloat16

__device__ __forceinline__ ortc::BFloat16 operator+(const ortc::BFloat16& a, const ortc::BFloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

__device__ __forceinline__ ortc::BFloat16 operator-(const ortc::BFloat16& a, const ortc::BFloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

__device__ __forceinline__ ortc::BFloat16 operator*(const ortc::BFloat16& a, const ortc::BFloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

__device__ __forceinline__ ortc::BFloat16 operator/(const ortc::BFloat16& a, const ortc::BFloat16& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

__device__ __forceinline__ ortc::BFloat16 operator-(const ortc::BFloat16& a) { return -static_cast<float>(a); }

__device__ __forceinline__ ortc::BFloat16& operator+=(ortc::BFloat16& a, const ortc::BFloat16& b) {
  a = a + b;
  return a;
}

__device__ __forceinline__ ortc::BFloat16& operator-=(ortc::BFloat16& a, const ortc::BFloat16& b) {
  a = a - b;
  return a;
}

__device__ __forceinline__ ortc::BFloat16& operator*=(ortc::BFloat16& a, const ortc::BFloat16& b) {
  a = a * b;
  return a;
}

__device__ __forceinline__ ortc::BFloat16& operator/=(ortc::BFloat16& a, const ortc::BFloat16& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

__device__ __forceinline__ float operator+(ortc::BFloat16 a, float b) { return a + b; }
__device__ __forceinline__ float operator-(ortc::BFloat16 a, float b) { return a - b; }
__device__ __forceinline__ float operator*(ortc::BFloat16 a, float b) { return a * b; }
__device__ __forceinline__ float operator/(ortc::BFloat16 a, float b) { return a / b; }

__device__ __forceinline__ float operator+(float a, ortc::BFloat16 b) { return a + b; }
__device__ __forceinline__ float operator-(float a, ortc::BFloat16 b) { return a - b; }
__device__ __forceinline__ float operator*(float a, ortc::BFloat16 b) { return a * b; }
__device__ __forceinline__ float operator/(float a, ortc::BFloat16 b) { return a / b; }

__device__ __forceinline__ float& operator+=(float& a, const ortc::BFloat16& b) { return a += b; }
__device__ __forceinline__ float& operator-=(float& a, const ortc::BFloat16& b) { return a -= b; }
__device__ __forceinline__ float& operator*=(float& a, const ortc::BFloat16& b) { return a *= b; }
__device__ __forceinline__ float& operator/=(float& a, const ortc::BFloat16& b) { return a /= b; }

/// Arithmetic with doubles

__device__ __forceinline__ double operator+(ortc::BFloat16 a, double b) { return static_cast<double>(a) + b; }
__device__ __forceinline__ double operator-(ortc::BFloat16 a, double b) { return static_cast<double>(a) - b; }
__device__ __forceinline__ double operator*(ortc::BFloat16 a, double b) { return static_cast<double>(a) * b; }
__device__ __forceinline__ double operator/(ortc::BFloat16 a, double b) { return static_cast<double>(a) / b; }

__device__ __forceinline__ double operator+(double a, ortc::BFloat16 b) { return a + static_cast<double>(b); }
__device__ __forceinline__ double operator-(double a, ortc::BFloat16 b) { return a - static_cast<double>(b); }
__device__ __forceinline__ double operator*(double a, ortc::BFloat16 b) { return a * static_cast<double>(b); }
__device__ __forceinline__ double operator/(double a, ortc::BFloat16 b) { return a / static_cast<double>(b); }

// Overloading < and > operators

__device__ __forceinline__ bool operator==(ortc::BFloat16& lhs, ortc::BFloat16& rhs) { return float(lhs) == float(rhs); }
__device__ __forceinline__ bool operator!=(ortc::BFloat16& lhs, ortc::BFloat16& rhs) { return float(lhs) != float(rhs); }
__device__ __forceinline__ bool operator>(ortc::BFloat16& lhs, ortc::BFloat16& rhs) { return float(lhs) > float(rhs); }
__device__ __forceinline__ bool operator<(ortc::BFloat16& lhs, ortc::BFloat16& rhs) { return float(lhs) < float(rhs); }

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
__device__ __inline__ ortc::BFloat16 _Tanh(ortc::BFloat16 a) { return tanhf(static_cast<float>(a)); }

inline OrtStatusPtr CudaCall(cudaError_t cuda_error) {
  if (cuda_error == cudaSuccess) return nullptr;
  return OrtW::API::CreateStatus(ORT_FAIL, MakeString("cuda error:", (int)cuda_error).c_str());
}
