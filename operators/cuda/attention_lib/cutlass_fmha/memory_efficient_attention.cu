// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if OCOS_USE_MEMORY_EFFICIENT_ATTENTION

#include "memory_efficient_attention.h"
#include <cassert>

namespace ort_extensions {
namespace cuda {

void run_memory_efficient_attention(const MemoryEfficientAttentionParams& params) {
  const int32_t& sm = params.sm;
  if (sm >= 80) {
    run_memory_efficient_attention_sm80(params);
  } else if (sm >= 75) {
    run_memory_efficient_attention_sm75(params);
  } else if (sm >= 70) {
    run_memory_efficient_attention_sm70(params);
  } else if (sm >= 50) {
    run_memory_efficient_attention_sm50(params);
  } else {
    assert(false);  // shall not reach here.
  }
}

}  // namespace cuda
}  // namespace ort_extensions

#endif  // OCOS_USE_MEMORY_EFFICIENT_ATTENTION
