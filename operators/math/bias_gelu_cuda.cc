// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_gelu_cuda.hpp"
#include "bias_gelu_cuda_impl.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

void bias_gelu_cuda(const ortc::Tensor<float>& X,
                    const ortc::Tensor<float>& B,
                    ortc::Tensor<float>& Y) {
    auto x_shape = X.Shape();
    launch_cuda_kernel(0UL, X.NumberOfElement(), B.NumberOfElement(), X.Data(), B.Data(), Y.Allocate(x_shape));
}
