// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "mul_sigmoid_impl.cuh"
#include "ortx_common.h"

namespace contrib {

template <typename T>
struct MulSigmoid {
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
    LaunchMulSigmoidKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                             input_length,
                             input_data,
                             output_data);
    return {};
  }
};

}  // namespace contrib