// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "mul_sigmoid_impl.cuh"

namespace contrib {

/**
* MulSigmoid(X) = X * Sigmoid(X)

No shape broadcasting supported.
*/
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

/**
* MulSigmoid(X, Y) = X * Y * Sigmoid(Y)

No shape broadcasting supported.
*/
template <typename T>
struct MulMulSigmoid {
  template <typename TDict>
  OrtxStatus OnModelAttach(const TDict& /*dict*/) {
    return {};
  }
  OrtxStatus Compute(Ort::Custom::CUDAKernelContext* ctx,
                       const ortc::Tensor<T>& input_x,
                       const ortc::Tensor<T>& input_y,
                       ortc::Tensor<T>& output) const {
    const T* input_data_x = input_x.Data();
    const T* input_data_y = input_y.Data();
    auto input_length_x = input_x.NumberOfElement();
    auto input_length_y = input_y.NumberOfElement();
    if (0 == input_length_x || 0 == input_data_y) {
      return {};
    }
    T* output_data = output.Allocate(input_length_x > input_length_y ? input_x.Shape() : input_y.Shape());
    LaunchMulMulSigmoidKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                                 input_length_x,
                                 input_length_y,
                                 input_data_x,
                                 input_data_y,
                                 output_data);
    return {};
  }
};

}  // namespace contrib
