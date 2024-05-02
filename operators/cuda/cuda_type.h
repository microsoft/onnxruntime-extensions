// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_fp16.h>
#include "onnxruntime_f16.h"
namespace contrib {
template <typename T>
struct CudaT {
  using MappedType = T;
};

template <>
struct CudaT<ortc::MFloat16> {
  using MappedType = half;
};
}
