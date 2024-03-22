// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_MEMORY_EFFICIENT_ATTENTION

#include "fmha_launch_template.h"

namespace contrib {
namespace cuda {

void run_memory_efficient_attention_sm50(const MemoryEfficientAttentionParams& params) {
  if (params.is_half) {
    DispatchBlockSize<cutlass::half_t, cutlass::arch::Sm50>(params);
  } else {
    DispatchBlockSize<float, cutlass::arch::Sm50>(params);
  }
}

}  // namespace cuda
}  // namespace contrib

#endif  // USE_MEMORY_EFFICIENT_ATTENTION
