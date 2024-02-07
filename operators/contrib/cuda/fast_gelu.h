// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "fast_gelu_impl.cuh"
#include <cuda_fp16.h>
#include <cublas_v2.h>
//#include "41_fused_multi_head_attention/kernel_forward.h"
#include "cute/arch/copy_sm90_desc.hpp"

namespace contrib {

template <typename T>
struct CudaT {
  using MappedType = T;
};

template <>
struct CudaT<ortc::MFloat16> {
  using MappedType = half;
};

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;

template <typename T>
inline IAllocatorUniquePtr<T> GetScrachBuffer(void* p, OrtAllocator* allocator) {
  return IAllocatorUniquePtr<T>{static_cast<T*>(p), [allocator = std::move(allocator)](T* p) {
    allocator->Free(allocator, p);
  }};
}

template <typename T>
struct FastGelu {
  OrtStatusPtr OnModelAttach(const OrtApi& /*api*/,
                             const OrtKernelInfo& /*info*/) {
    return nullptr;
  }
  OrtStatusPtr Compute(OrtKernelContext* kernel_context,
                       const Ort::Custom::CudaContext& ctx,
                       const ortc::Tensor<T>& input,
                       std::optional<const ortc::Tensor<T>*> bias,
                       ortc::Tensor<T>& output) const {
    if (kernel_context == nullptr) return nullptr;
    size_t input_count = 0;
    auto hr = OrtW::API::GetInputCount(kernel_context, &input_count);
    if (hr || input_count == 0) return nullptr;
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(ctx.cublas);
    if (!cublas) return nullptr;
    //bool supportsDropout = AttentionKernel::kSupportsDropout;
    auto value = cute::TMA::SmemSwizzleBits::B32;
    OrtMemoryInfo* mem_info = OrtW::API::CreateOrtMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    OrtAllocator* allocator = OrtW::API::GetOrtAllocator(kernel_context, mem_info);
    void* p_raw = allocator->Alloc(allocator, 3);
    if (!p_raw) return nullptr;
    {
      IAllocatorUniquePtr<int> p = GetScrachBuffer<int>(p_raw, allocator);
    }

    const T* input_data = input.Data();
    T* output_data = output.Allocate(input.Shape());
    auto input_length = input.NumberOfElement();
    if (0 == input_length) {
      return nullptr;
    }
    const T* bias_data = bias.has_value() ? (*bias)->Data() : nullptr;
    auto bias_length = bias.has_value() ? (*bias)->NumberOfElement() : 0;
    using TT = typename CudaT<T>::MappedType;
    LaunchFastGeluKernel<TT>(reinterpret_cast<cudaStream_t>(ctx.cuda_stream),
                             input_length,
                             bias_length,
                             reinterpret_cast<const TT*>(input_data),
                             reinterpret_cast<const TT*>(bias_data),
                             reinterpret_cast<TT*>(output_data),
                             use_half2_);
    return nullptr;
  }

 private:
  bool use_half2_ = false;  // to-do, read this from env var
};

}  // namespace contrib