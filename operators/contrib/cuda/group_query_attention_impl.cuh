// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cublas_v2.h>
#include "onnxruntime_c_api.h"
#include "../attention_common.h"

namespace contrib {
namespace cuda {

template <typename T>
struct GroupQueryAttentionData {
  // Input Tensors
  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  const T* past_key = nullptr;
  const T* past_value = nullptr;
  int* seqlens_k = nullptr;
  const T* cos_cache = nullptr;
  const T* sin_cache = nullptr;
  // Flash buffers
  T* softmax_lse = nullptr;
  T* softmax_lse_accum = nullptr;
  T* out_accum = nullptr;
  int* seqlens_k_total = nullptr;
  // Memory Efficient buffers
  T* fmha_buffer = nullptr;
  T* k = nullptr;
  T* v = nullptr;
  // Output Tensors
  T* output = nullptr;
  T* present_key = nullptr;
  T* present_value = nullptr;
  // Kernel Flags
  bool use_flash_attention = false;
  bool use_memory_efficient_attention = false;
};

template <typename T>
OrtStatusPtr QkvToContext(
//    const cudaDeviceProp& device_prop,
//    cublasHandle_t& cublas, // TODO: cublas is not used at all
    cudaStream_t cuda_stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T>& data);

}  // namespace cuda
}  // namespace contrib
