// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"

#ifdef USE_CUDA
#include "cuda/add_mul.h"
#include "cuda/fast_gelu.h"
#include "cuda/mul_sigmoid.h"
#include "cuda/negxplus1.h"
#include "cuda/replace_zero.h"
#include "cuda/scatter_nd_of_shape.h"
#include "cuda/transpose_cast.h"
#endif

FxLoadCustomOpFactory LoadCustomOpClasses_Contrib = []() -> CustomOpArray& {
  using AddSharedInputFloat32Type = typename contrib::AddOrMulSharedInput<float, true>;
  using MulSharedInputFloat32Type = typename contrib::AddOrMulSharedInput<float, false>;

#if ORT_API_VERSION >= 16
  using AddSharedInputFloat16Type = typename contrib::AddOrMulSharedInput<ortc::MFloat16, true>;
  using MulSharedInputFloat16Type = typename contrib::AddOrMulSharedInput<ortc::MFloat16, false>;
  using Transpose2DCastFloat32ToFloat16Type = typename contrib::Transpose2DCast<float, ortc::MFloat16>;
  using Transpose2DCastFloat16ToFloat32Type = typename contrib::Transpose2DCast<ortc::MFloat16, float>;
#endif

  static OrtOpLoader op_loader(
      []() { return nullptr; }
#ifdef USE_CUDA
      ,
      CustomCudaStructV2("AddSharedInput", AddSharedInputFloat32Type),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<float>),
      CustomCudaStructV2("MaskedScatterNDOfShape", contrib::MaskedScatterNDOfShape<float>),
      CustomCudaStructV2("MulSharedInput", MulSharedInputFloat32Type),
      CustomCudaStructV2("MulMulSigmoid", contrib::MulMulSigmoid<float>),
      CustomCudaStructV2("MulSigmoid", contrib::MulSigmoid<float>),
      CustomCudaStructV2("NegXPlus1", contrib::NegXPlus1<float>),
      CustomCudaStructV2("ReplaceZero", contrib::ReplaceZero<float>),
      CustomCudaStructV2("ScatterNDOfShape", contrib::ScatterNDOfShape<float>),
#if ORT_API_VERSION >= 16

      CustomCudaStructV2("AddSharedInput", AddSharedInputFloat16Type),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::MFloat16>),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::BFloat16>),
      CustomCudaStructV2("MaskedScatterNDOfShape", contrib::MaskedScatterNDOfShape<ortc::MFloat16>),
      CustomCudaStructV2("MulSharedInput", MulSharedInputFloat16Type),
      CustomCudaStructV2("MulMulSigmoid", contrib::MulMulSigmoid<ortc::MFloat16>),
      CustomCudaStructV2("MulSigmoid", contrib::MulSigmoid<ortc::MFloat16>),
      CustomCudaStructV2("NegXPlus1", contrib::NegXPlus1<ortc::MFloat16>),
      CustomCudaStructV2("ReplaceZero", contrib::ReplaceZero<ortc::MFloat16>),
      CustomCudaStructV2("ScatterNDOfShape", contrib::ScatterNDOfShape<ortc::MFloat16>),
      CustomCudaStructV2("Transpose2DCastFP16", Transpose2DCastFloat32ToFloat16Type),
      CustomCudaStructV2("Transpose2DCastFP32", Transpose2DCastFloat16ToFloat32Type)
#endif
#endif
  );
  return op_loader.GetCustomOps();
};
