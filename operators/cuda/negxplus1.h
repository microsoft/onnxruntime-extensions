// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "negxplus1_impl.cuh"

namespace contrib {

template <typename T>
struct NegXPlus1 {
  template <typename TDict>
  OrtStatusPtr OnModelAttach(const TDict& /*dict*/) {
    return nullptr;
  }
  OrtStatusPtr Compute(Ort::Custom::CUDAKernelContext* ctx,
                       const ortc::Tensor<T>& input,
                       ortc::Tensor<T>& output) const {
    const T* input_data = input.Data();
    T* output_data = output.Allocate(input.Shape());
    auto input_length = input.NumberOfElement();
    if (0 == input_length) {
      return nullptr;
    }
    LaunchNegXPlus1Kernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                             input_length,
                             input_data,
                             output_data);
    return nullptr;
  }
};

}  // namespace contrib