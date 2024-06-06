// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "transpose_cast_impl.cuh"
#include "ortx_common.h"

namespace contrib {

template <typename TIN, typename TOUT>
struct Transpose2DCast {
  template <typename TDict>
  OrtxStatus OnModelAttach(const TDict& /*dict*/) {
    return {};
  }
  OrtxStatus Compute(Ort::Custom::CUDAKernelContext* ctx,
                       const ortc::Tensor<TIN>& input,
                       ortc::Tensor<TOUT>& output) const {
    const TIN* input_data = input.Data();
    auto shape = input.Shape();
    if (shape.size() != 2) {
      ORTX_CXX_API_THROW("Input must be a 2D tensor", ORT_RUNTIME_EXCEPTION);
    }
    int n_rows = static_cast<int>(shape[0]);
    int n_cols = static_cast<int>(shape[1]);

    std::vector<int64_t> new_shape{static_cast<int64_t>(n_cols), static_cast<int64_t>(n_rows)};
    TOUT* output_data = output.Allocate(new_shape);
    if (0 == n_rows || 0 == n_cols) {
      return {};
    }
    LaunchTranspose2DCastKernel<TIN, TOUT>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                                           n_rows, n_cols, input_data, output_data);
    return {};
  }
};

}  // namespace contrib