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
        : lenght_a <= length_b ? tensor_b.Shape()
                               : tensor_a.Shape());

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

template <typename T, bool addition_first>
struct AddAndMul {
  template <typename TDict>
  OrtxStatus OnModelAttach(const TDict& dict) {
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
    if (0 == input_data_a || 0 == input_data_b || 0 == input_data_c) {
      return {};
    }

    std::vector<int64_t> dimsA = tensor_a.Shape();
    std::vector<int64_t> dimsB = tensor_b.Shape();
    std::vector<int64_t> dimsC = tensor_c.Shape();

    auto max_length = std::max(length_a, std::max(length_b, length_c));

    auto max_rank = std::max(dimsA.size(), std::max(dimsB.size(), dimsC.size()));
    while (dimsA.size() < max_rank)
      dimsA.insert(dimsA.begin(), 1);
    while (dimsB.size() < max_rank)
      dimsB.insert(dimsB.begin(), 1);
    while (dimsC.size() < max_rank)
      dimsC.insert(dimsC.begin(), 1);

    std::vector<int64_t> output_dims(dimsA.size());
    for (size_t i = 0; i < dimsA.size(); ++i) {
      output_dims[i] = std::max(std::max(dimsA[i], dimsB[i]), dimsC[i]);
    }

    if (switchMiddelAxis_) {
      if (output_dims.size() != 4) {
        ORTX_CXX_API_THROW("switchMiddleAxes only works with 4D tensors", ORT_RUNTIME_EXCEPTION);
      }
      int64_t d4 = output_dims[output_dims.size() - 1];
      int64_t d3 = output_dims[output_dims.size() - 2];
      int64_t d2 = output_dims[output_dims.size() - 3];
      output_dims[1] = d3;
      output_dims[2] = d2;
      LaunchAddAndMulSwitchMiddleAxesKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                                               input_data_a, input_data_b, input_data_c,
                                               output_data,
                                               length_a, length_b, length_c,
                                               addition_first, d2, d3, d4);
    } else {
      T* output_data_ab = output_ab.Allocate(output_dims);
      LaunchAddAndMulKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                               input_data_a, input_data_b, input_data_c,
                               output_data,
                               length_a, length_b, length_c,
                               addition_first);
    }
    return {};
  }

 private:
  bool switchMiddelAxis_;
};

template <typename T, bool subtract_first>
struct SubAndMul {
  template <typename TDict>
  OrtxStatus OnModelAttach(const TDict& dict) {
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
    if (0 == input_data_a || 0 == input_data_b || 0 == input_data_c) {
      return {};
    }

    std::vector<int64_t> dimsA = tensor_a.Shape();
    std::vector<int64_t> dimsB = tensor_b.Shape();
    std::vector<int64_t> dimsC = tensor_c.Shape();

    auto max_length = std::max(length_a, std::max(length_b, length_c));

    auto max_rank = std::max(dimsA.size(), std::max(dimsB.size(), dimsC.size()));
    while (dimsA.size() < max_rank)
      dimsA.insert(dimsA.begin(), 1);
    while (dimsB.size() < max_rank)
      dimsB.insert(dimsB.begin(), 1);
    while (dimsC.size() < max_rank)
      dimsC.insert(dimsC.begin(), 1);

    std::vector<int64_t> output_dims(dimsA.size());
    for (size_t i = 0; i < dimsA.size(); ++i) {
      output_dims[i] = std::max(std::max(dimsA[i], dimsB[i]), dimsC[i]);
    }

    T* output_data_ab = output_ab.Allocate(output_dims);
    LaunchSubAndMulKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                             input_data_a, input_data_b, input_data_c,
                             output_data,
                             length_a, length_b, length_c,
                             subtract_first, negative_);
    return {};
  }

 private:
  bool negative_;
};

}  // namespace contrib