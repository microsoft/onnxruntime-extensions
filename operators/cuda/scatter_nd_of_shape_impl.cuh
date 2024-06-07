// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

namespace contrib {

enum class ScatterReduction : int {
  None = 0,
  Add = 1,
  Mul = 2,
  Min = 3,
  Max = 4,
};

template <typename T>
cudaError_t LaunchScatterNDOfShapeKernel(cudaStream_t stream,
                                         const std::vector<int64_t>& output_shape,
                                         const std::vector<int64_t>& indices_shape,
                                         const int64_t* indices,
                                         const T* updates,
                                         T* output,
                                         ScatterReduction reduction);

template <typename T>
cudaError_t LaunchMaskedScatterNDOfShapeKernel(cudaStream_t stream,
                                               const std::vector<int64_t>& output_shape,
                                               const std::vector<int64_t>& indices_shape,
                                               const int64_t* indices,
                                               const T* updates,
                                               T* output,
                                               ScatterReduction reduction,
                                               int64_t masked_value);

}  // namespace contrib
