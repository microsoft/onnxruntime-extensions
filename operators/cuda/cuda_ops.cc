// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"

#ifdef USE_CUDA
#include "cuda/fast_gelu.h"
#include "cuda/negxplus1.h"
#include "cuda/scatter_nd_of_shape.h"
#endif

FxLoadCustomOpFactory LoadCustomOpClasses_Contrib = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(
      []() { return nullptr; }
#ifdef USE_CUDA
      ,
      CustomCudaStructV2("FastGelu", contrib::FastGelu<float>),
      CustomCudaStructV2("MaskedScatterNDOfShape", contrib::MaskedScatterNDOfShape<float>),
      CustomCudaStructV2("NegXPlus1", contrib::NegXPlus1<float>),
      CustomCudaStructV2("ScatterNDOfShape", contrib::ScatterNDOfShape<float>),
#if ORT_API_VERSION >= 16

      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::MFloat16>),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::BFloat16>),
      CustomCudaStructV2("MaskedScatterNDOfShape", contrib::MaskedScatterNDOfShape<ortc::MFloat16>),
      CustomCudaStructV2("NegXPlus1", contrib::NegXPlus1<ortc::MFloat16>),
      CustomCudaStructV2("ScatterNDOfShape", contrib::ScatterNDOfShape<ortc::MFloat16>)
#endif
#endif
  );
  return op_loader.GetCustomOps();
};
