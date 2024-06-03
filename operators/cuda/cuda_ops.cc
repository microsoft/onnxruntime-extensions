// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"

#ifdef USE_CUDA
#include "cuda/add_mul.h"
#include "cuda/fast_gelu.h"
#include "cuda/negxplus1.h"
#endif

FxLoadCustomOpFactory LoadCustomOpClasses_Contrib = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(
      []() { return nullptr; }
#ifdef USE_CUDA
      ,
      CustomCudaStructV2("AddSharedInput", contrib::AddOrMulSharedInput<float, true>),
      CustomCudaStructV2("MulSharedInput", contrib::AddOrMulSharedInput<float, false>),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<float>),
      CustomCudaStructV2("NegXPlus1", contrib::NegXPlus1<float>),
#if ORT_API_VERSION >= 16

      CustomCudaStructV2("AddSharedInput", contrib::AddOrMulSharedInput<ortc::MFloat16, true>),
      CustomCudaStructV2("MulSharedInput", contrib::AddOrMulSharedInput<ortc::MFloat16, false>),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::MFloat16>),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::BFloat16>),
      CustomCudaStructV2("NegXPlus1", contrib::NegXPlus1<ortc::MFloat16>)
#endif
#endif
  );
  return op_loader.GetCustomOps();
};
