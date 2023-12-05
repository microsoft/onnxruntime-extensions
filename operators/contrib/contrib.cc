// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"

#ifdef USE_CUDA
#include "cuda/fast_gelu.h"
#endif

FxLoadCustomOpFactory LoadCustomOpClasses_Contrib = []() -> CustomOpArray& {
#ifdef USE_CUDA
  static OrtOpLoader op_loader(CustomCudaStructV2("FastGelu", contrib::FastGelu<float>));
#else
  static OrtOpLoader op_loader;
#endif
  return op_loader.GetCustomOps();
};
