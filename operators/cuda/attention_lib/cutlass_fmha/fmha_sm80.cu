// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if OCOS_USE_MEMORY_EFFICIENT_ATTENTION

#include "fmha_launch_template.h"

namespace ort_extensions {
namespace cuda {

void run_memory_efficient_attention_sm80(const MemoryEfficientAttentionParams& params) {
  if (params.is_half) {
    DispatchBlockSize<cutlass::half_t, cutlass::arch::Sm80>(params);
  } else {
    DispatchBlockSize<float, cutlass::arch::Sm80>(params);
  }
}

}  // namespace cuda
}  // namespace ort_extensions

#endif  // OCOS_USE_MEMORY_EFFICIENT_ATTENTION
