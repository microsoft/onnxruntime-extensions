// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"

#ifdef USE_CUDA
#include "cuda/fast_gelu.h"
#include "cuda/scatter_nd_of_shape.h"
#endif

FxLoadCustomOpFactory LoadCustomOpClasses_Contrib = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(
      []() { return nullptr; }
#ifdef USE_CUDA
      ,
      CustomCudaStructV2("FastGelu", contrib::FastGelu<float>),
#if ORT_API_VERSION >= 16

      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::MFloat16>),
      CustomCudaStructV2("FastGelu", contrib::FastGelu<ortc::BFloat16>),
      []() { return std::shared_ptr<OrtCustomOp>(std::make_unique<contrib::ScatterNDOfShapeOp<float>>().release()); },
      []() { return std::shared_ptr<OrtCustomOp>(std::make_unique<contrib::ScatterNDOfShapeOp<ortc::MFloat16>>().release()); }
#endif
#endif
  );
  return op_loader.GetCustomOps();
};
