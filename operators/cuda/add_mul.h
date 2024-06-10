// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "add_mul_impl.cuh"
#include "ortx_common.h"

namespace contrib {

inline void _FillOutputShape3Op(std::vector<int64_t>& dimsA,
                                std::vector<int64_t>& dimsB,
                                std::vector<int64_t>& dimsC,
                                std::vector<int64_t>& output_dims) {
    auto max_rank = std::max(dimsA.size(), std::max(dimsB.size(), dimsC.size()));
    while (dimsA.size() < max_rank)
      dimsA.insert(dimsA.begin(), 1);
    while (dimsB.size() < max_rank)
      dimsB.insert(dimsB.begin(), 1);
    while (dimsC.size() < max_rank)
      dimsC.insert(dimsC.begin(), 1);

    output_dims.resize(dimsA.size());
    for (size_t i = 0; i < dimsA.size(); ++i) {
      output_dims[i] = std::max(std::max(dimsA[i], dimsB[i]), dimsC[i]);
      if (output_dims[i] == 0) {
        ORTX_CXX_API_THROW("One of the input dimensions is null.", ORT_RUNTIME_EXCEPTION);        
      }
    }
}

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
    auto length_a = tensor_a.NumberOfElement();
    auto length_b = tensor_b.NumberOfElement();
    auto length_c = tensor_c.NumberOfElement();

    if (0 == length_a || 0 == length_b || 0 == length_c) {
      return {};
    }

    T* output_data_ab = output_ab.Allocate(length_a <= length_b ? tensor_b.Shape() : tensor_a.Shape());
    T* output_data_ac = output_ac.Allocate(length_a <= length_c ? tensor_c.Shape() : tensor_a.Shape());

    LaunchAddOrMulSharedInputKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                                       tensor_a.Data(), tensor_b.Data(), tensor_c.Data(),
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
    auto length_a = tensor_a.NumberOfElement();
    auto length_b = tensor_b.NumberOfElement();
    auto length_c = tensor_c.NumberOfElement();

    if (0 == length_a || 0 == length_b || 0 == length_c) {
      return {};
    }

    std::vector<int64_t> dimsA = tensor_a.Shape();
    std::vector<int64_t> dimsB = tensor_b.Shape();
    std::vector<int64_t> dimsC = tensor_c.Shape();
    std::vector<int64_t> output_dims;
    _FillOutputShape3Op(dimsA, dimsB, dimsC, output_dims);

    T* output_data = output.Allocate(output_dims);

    LaunchAddOrMulTwiceKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                                 tensor_a.Data(), tensor_b.Data(), tensor_c.Data(),
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
    int64_t default_value = 0;
    switchMiddelAxis_ = dict.TryToGetAttributeWithDefault("switchMiddleAxis", default_value) == 1;
  }
  OrtxStatus Compute(Ort::Custom::CUDAKernelContext* ctx,
                     const ortc::Tensor<T>& tensor_a,
                     const ortc::Tensor<T>& tensor_b,
                     const ortc::Tensor<T>& tensor_c,
                     ortc::Tensor<T>& output) const {
    auto length_a = tensor_a.NumberOfElement();
    auto length_b = tensor_b.NumberOfElement();
    auto length_c = tensor_c.NumberOfElement();

    if (0 == length_a || 0 == length_b || 0 == length_c) {
      return {};
    }

    std::vector<int64_t> dimsA = tensor_a.Shape();
    std::vector<int64_t> dimsB = tensor_b.Shape();
    std::vector<int64_t> dimsC = tensor_c.Shape();
    std::vector<int64_t> output_dims;
    _FillOutputShape3Op(dimsA, dimsB, dimsC, output_dims);

    if (switchMiddelAxis_) {
      if (output_dims.size() != 4) {
        ORTX_CXX_API_THROW("switchMiddleAxes only works with 4D tensors", ORT_RUNTIME_EXCEPTION);
      }
      int64_t d4 = output_dims[output_dims.size() - 1];
      int64_t d3 = output_dims[output_dims.size() - 2];
      int64_t d2 = output_dims[output_dims.size() - 3];
      output_dims[1] = d3;
      output_dims[2] = d2;
      T* output_data = output.Allocate(output_dims);
      LaunchAddAndMulSwitchMiddleAxesKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                                               tensor_a.Data(), tensor_b.Data(), tensor_c.Data(),
                                               output_data,
                                               length_a, length_b, length_c,
                                               addition_first, d2, d3, d4);
    } else {
      T* output_data = output.Allocate(output_dims);
      LaunchAddAndMulKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                               tensor_a.Data(), tensor_b.Data(), tensor_c.Data(),
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
    //int64_t default_value = 0;
    //negative_ = dict.TryToGetAttributeWithDefault("negative", default_value) == 1;
    negative_ = false;
  }
  OrtxStatus Compute(Ort::Custom::CUDAKernelContext* ctx,
                     const ortc::Tensor<T>& tensor_a,
                     const ortc::Tensor<T>& tensor_b,
                     const ortc::Tensor<T>& tensor_c,
                     ortc::Tensor<T>& output) const {
    auto length_a = tensor_a.NumberOfElement();
    auto length_b = tensor_b.NumberOfElement();
    auto length_c = tensor_c.NumberOfElement();
    if (0 == length_a || 0 == length_b || 0 == length_c) {
      return {};
    }

    std::vector<int64_t> dimsA = tensor_a.Shape();
    std::vector<int64_t> dimsB = tensor_b.Shape();
    std::vector<int64_t> dimsC = tensor_c.Shape();
    std::vector<int64_t> output_dims;
    _FillOutputShape3Op(dimsA, dimsB, dimsC, output_dims);
    T* output_data = output.Allocate(output_dims);

    LaunchSubAndMulKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                             tensor_a.Data(), tensor_b.Data(), tensor_c.Data(),
                             output_data,
                             length_a, length_b, length_c,
                             subtract_first, negative_);
    return {};
  }

 private:
  bool negative_;
};

}  // namespace contrib