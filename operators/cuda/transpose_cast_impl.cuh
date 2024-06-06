// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

template <typename TIN, typename TOUT>
cudaError_t LaunchTranspose2DCastKernel(cudaStream_t stream, int n_rows, int n_cols, const TIN* input, TOUT* output);