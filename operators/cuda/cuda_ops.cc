// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"

#ifdef USE_CUDA
#include "cuda/add_mul.h"
#include "cuda/fast_gelu.h"
#include "cuda/negxplus1.h"
#endif

FxLoadCustomOpFactory LoadCustomOpClasses_Contrib = []() -> CustomOpArray& {

  using AddSharedInputFloat32Type = typename contrib::AddOrMulSharedInput<float, true>;
  using MulSharedInputFloat32Type = typename contrib::AddOrMulSharedInput<float, false>;

#if ORT_API_VERSION >= 16
  using AddSharedInputFloat16Type = typename contrib::AddOrMulSharedInput<ortc::MFloat16, true>;
  using MulSharedInputFloat16Type = typename contrib::AddOrMulSharedInput<ortc::MFloat16, false>;
#endif


  static OrtOpLoader op_loader(
      []() { return nullptr; }
#ifdef USE_CUDA
      ,
      CustomCudaStructV2("AddSharedInput", AddSharedInputFloat32Type),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<float>),
      CustomCudaStructV2("MulSharedInput", MulSharedInputFloat32Type),
      CustomCudaStructV2("NegXPlus1", contrib::NegXPlus1<float>),
#if ORT_API_VERSION >= 16

      CustomCudaStructV2("AddSharedInput", AddSharedInputFloat16Type),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::MFloat16>),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::BFloat16>),
      CustomCudaStructV2("MulSharedInput", MulSharedInputFloat16Type),
      CustomCudaStructV2("NegXPlus1", contrib::NegXPlus1<ortc::MFloat16>)
#endif
#endif
  );
  return op_loader.GetCustomOps();
};
