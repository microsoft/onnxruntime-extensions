// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"

#ifdef USE_CUDA
#include "cuda/fast_gelu.h"
#if ORT_API_VERSION >= 18
#include "cuda/paged_attention.h"
#endif
#endif

FxLoadCustomOpFactory LoadCustomOpClasses_Contrib = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(
      []() { return nullptr; }
#ifdef USE_CUDA
      ,
      CustomCudaStructV2("FastGelu", contrib::FastGelu<float>),
#if ORT_API_VERSION >= 18
      CustomCudaStructV2("PagedAttention", PagedAttention<ortc::MFloat16>),
#endif
#if ORT_API_VERSION >= 16
      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::MFloat16>),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::BFloat16>)
#endif
#endif
  );
  return op_loader.GetCustomOps();
};
