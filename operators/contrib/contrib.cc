// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"

#ifdef USE_CUDA
#include "cuda/fast_gelu.h"
#endif

FxLoadCustomOpFactory LoadCustomOpClasses_Contrib = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(
    []() { return nullptr; }
#ifdef USE_CUDA
    ,
    CustomCudaStructV2("FastGelu", contrib::FastGelu<float>)
#endif
    );
  return op_loader.GetCustomOps();
};

