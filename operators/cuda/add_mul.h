// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "add_mul_impl.cuh"
#include "ortx_common.h"

namespace contrib {

template <typename T, bool addition>
struct AddOrMulSharedInput {
  template <typename TDict>
  OrtxStatus OnModelAttach(const TDict& /*dict*/) {
    return {};
  }
  OrtxStatus Compute(Ort::Custom::CUDAKernelContext* ctx,
                     const ortc::Tensor<T>& tensor_a,
                     const ortc::Tensor<T>& tensor_b,
                     const ortc::Tensor<T>& tensor_c,
                     ortc::Tensor<T>& output_ab,
                     ortc::Tensor<T>& output_ac) const {
    const T* input_data_a = tensor_a.Data();
    const T* input_data_b = tensor_b.Data();
    const T* input_data_c = tensor_c.Data();

    auto length_a = tensor_a.NumberOfElement();
    auto length_b = tensor_b.NumberOfElement();
    auto length_c = tensor_c.NumberOfElement();

    T* output_data_ab = output_ab.Allocate(length_a <= length_b ? tensor_b.Shape() : tensor_a.Shape());
    T* output_data_ac = output_ab.Allocate(length_a <= length_c ? tensor_c.Shape() : tensor_a.Shape());

    if (0 == input_data_a || 0 == input_data_b || 0 == input_data_c) {
      return {};
    }
    LaunchAddOrMulSharedInputKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                                       input_data_a, input_data_b, input_data_c,
                                       output_data_ab, output_data_ac,
                                       length_a, length_b, length_c,
                                       addition);
    return {};
  }
};

template <typename T, bool addition>
struct AddOrMulTwice {
  template <typename TDict>
  OrtxStatus OnModelAttach(const TDict& /*dict*/) {
    return {};
  }
  OrtxStatus Compute(Ort::Custom::CUDAKernelContext* ctx,
                     const ortc::Tensor<T>& tensor_a,
                     const ortc::Tensor<T>& tensor_b,
                     const ortc::Tensor<T>& tensor_c,
                     ortc::Tensor<T>& output) const {
    const T* input_data_a = tensor_a.Data();
    const T* input_data_b = tensor_b.Data();
    const T* input_data_c = tensor_c.Data();

    auto length_a = tensor_a.NumberOfElement();
    auto length_b = tensor_b.NumberOfElement();
    auto length_c = tensor_c.NumberOfElement();

    T* output_data_ab = output_ab.Allocate(
      length_a <= length_b
        ? lenght_c <= length_b ? tensor_b.Shape() : tensor_c.Shape()
        : lenght_a <= length_b ? tensor_b.Shape() : tensor_a.Shape());

    if (0 == input_data_a || 0 == input_data_b || 0 == input_data_c) {
      return {};
    }
    LaunchAddOrMulTwiceKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                                       input_data_a, input_data_b, input_data_c,
                                       output_data,
                                       length_a, length_b, length_c,
                                       addition);
    return {};
  }
};


}  // namespace contrib