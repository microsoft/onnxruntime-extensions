// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "replace_zero_impl.cuh"

namespace contrib {

/**
* Y = ReplaceZero(X, by=c) is equivalent to:
*
* Y = X.copy()
* X[X == 0] = c
*
* This operation usually appears when a tensor is updated with an operator Equal and Where.
* This kernel avoids the creation of one null tensor.
*/
template <typename T>
struct ReplaceZero {
  template <typename TDict>
  OrtxStatus OnModelAttach(const TDict& dict) {
    float default_value=0;
    by_ = dict.TryToGetAttributeWithDefault("by", default_value);
    return {};
  }
  OrtxStatus Compute(Ort::Custom::CUDAKernelContext* ctx,
                       const ortc::Tensor<T>& input,
                       ortc::Tensor<T>& output) const {
    const T* input_data = input.Data();
    auto input_shape = input.Shape();
    T* output_data = output.Allocate(input_shape);
    auto input_length = input.NumberOfElement();
    if (0 == input_length) {
      return {};
    }

    LaunchReplaceZeroKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                             input_length,
                             input_data,
                             output_data,
                             by_);
    return {};
  }

  private:
  float by_;
};

}  // namespace contrib