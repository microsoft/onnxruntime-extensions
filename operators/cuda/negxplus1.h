// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "negxplus1_impl.cuh"
#include "ortx_common.h"

namespace contrib {

/**
* NegXPlus1(X) = 1 - X
*/
template <typename T>
struct NegXPlus1 {
  template <typename TDict>
  OrtxStatus OnModelAttach(const TDict& /*dict*/) {
    return {};
  }
  OrtxStatus Compute(Ort::Custom::CUDAKernelContext* ctx,
                       const ortc::Tensor<T>& input,
                       ortc::Tensor<T>& output) const {
    const T* input_data = input.Data();
    T* output_data = output.Allocate(input.Shape());
    auto input_length = input.NumberOfElement();
    if (0 == input_length) {
      return {};
    }
    LaunchNegXPlus1Kernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                             input_length,
                             input_data,
                             output_data);
    return {};
  }
};

}  // namespace contrib