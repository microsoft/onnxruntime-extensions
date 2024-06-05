// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

template <typename TIN, typename TOUT>
cudaError_t TransposeCast2DKernel(cudaStream_t stream, size_t n_rows, size_t n_cols, const TIN* input, TOUT* output);