// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "fast_gelu_impl.cuh"

namespace contrib {

template <typename T>
struct FastGelu {
  OrtStatusPtr OnModelAttach(const OrtApi& /*api*/,
                             const OrtKernelInfo& /*info*/) {
    return nullptr;
  }
  OrtStatusPtr Compute(const Ort::Custom::CudaContext& ctx,
                       const ortc::Tensor<T>& input,
                       std::optional<const ortc::Tensor<T>*> bias,
                       ortc::Tensor<T>& output) const {
    const T* input_data = input.Data();
    T* output_data = output.Allocate(input.Shape());
    auto input_length = input.NumberOfElement();
    if (0 == input_length) {
        return nullptr;
    }
    const T* bias_data = bias.has_value()?(*bias)->Data():nullptr;
    auto bias_length = bias.has_value()?(*bias)->NumberOfElement():0;
    LaunchFastGeluKernel(reinterpret_cast<cudaStream_t>(ctx.cuda_stream),
                         input_length,
                         bias_length,
                         input_data,
                         bias_data,
                         output_data,
                         use_half2_);
    return nullptr;
  }

 private:
  bool use_half2_ = false; // to-do, read this from env var
};

}