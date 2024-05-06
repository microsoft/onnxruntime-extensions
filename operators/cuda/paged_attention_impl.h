// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include <cuda.h>

template <typename T>
using UniquePtrWithDeletor = std::unique_ptr<T, std::function<void(T*)>>;

template <typename T>
inline UniquePtrWithDeletor<T> GetScratchBuffer(void* p, OrtAllocator* allocator) {
  return UniquePtrWithDeletor<T>{static_cast<T*>(p), [allocator = std::move(allocator)](T* p) {
                                  allocator->Free(allocator, p);
                                }};
}

namespace cuda {
struct InputMetadata {
  //int64_t schedule_type;  // 0: vllm. 1:sarathi, 2:custom, 3:self-build
  //int64_t block_tables;
  int64_t max_num_blocks_per_seq;
  //int64_t context_lens;
  int64_t max_context_len = 0;
  int64_t num_prompt_tokens = 0;
  int64_t num_valid_tokens = 0;
  //int64_t slot_mapping;
  int64_t num_generation_tokens = 0;

  UniquePtrWithDeletor<int64_t> position_ids; 
};

void InitializeHeadMapping();

// TODO(leca): remove unnecessary parameters
template <typename T>
OrtStatusPtr CheckInputs(const cudaStream_t stream, OrtAllocator* allocator, const ortc::Tensor<T>& query, const ortc::Tensor<T>& key,
                         const ortc::Tensor<T>& value, const ortc::Tensor<T>& key_cache, const ortc::Tensor<T>& value_cache,
                         const ortc::Tensor<int32_t>& block_tables, const ortc::Tensor<int32_t>& slot_mappings, 
                         std::optional<const ortc::Tensor<int32_t>*> context_lens,
                         std::optional<const ortc::Tensor<int64_t>*> positions, InputMetadata& input_metadata);

void paged_attention_v1(
    const cudaStream_t stream,
    void* out,                // [num_seqs, num_heads, head_size]
    const void* query,        // [num_seqs, num_heads, head_size]
    const void* key_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const void* value_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int* head_mapping,  // [num_heads]
    float scale,
    const int* block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* context_lens,  // [num_seqs]
    int block_size,
    int max_context_len,
    const float* __restrict__ alibi_slopes,
    const int max_num_blocks_per_seq,
    const int64_t* query_shapes,
    int num_queries_per_kv,
    int dtype);
//    const void* kv_quant_params_cache = nullptr,  // [num_blocks, 2, num_kv_heads, head_size / kv_quant_chunk_size, block_size]
//    int kv_quant_chunk_size = 0,
//    int kv_quant_param_dtype = 0);

void paged_attention_v2(
    const cudaStream_t stream,
    void* out,                // [num_seqs, num_heads, head_size]
    void* exp_sums,           // [num_seqs, num_heads, max_num_partitions]
    void* max_logits,         // [num_seqs, num_heads, max_num_partitions]
    void* tmp_out,            // [num_seqs, num_heads, max_num_partitions, head_size]
    const void* query,        // [num_seqs, num_heads, head_size]
    const void* key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    const void* value_cache,  // [num_blocks, num_heads, head_size, block_size]
    const int* head_mapping,  // [num_heads]
    float scale,
    const int* block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* context_lens,  // [num_seqs]
    int block_size,
    int max_context_len,
    const float* alibi_slopes,
    const int max_num_blocks_per_seq,
    const int64_t* query_shapes,
    int num_queries_per_kv,
    int dtype);

void reshape_and_cache(
    const cudaStream_t stream,
    const void* key,          // [num_tokens, num_heads, head_size]
    const void* value,        // [num_tokens, num_heads, head_size]
    const void* key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    const void* value_cache,  // [num_blocks, num_heads, head_size, block_size]
    const int* slot_mapping,  // [num_tokens]
    const int64_t* key_shapes,
    const int64_t* value_shapes,
    const int64_t block_size,
    const int vec_x,
    int dtype);
//    void* kv_quant_param = nullptr,  // [num_blocks, 2, num_heads, head_size / kv_quant_chunk_size, block_size]
//    const int kv_quant_chunk_size = 0,
//    const int kv_quant_param_dtype = 1);

void rotary_embedding_neox(
    const cudaStream_t stream,
    const int64_t* positions,  // [num_tokens]
    void* query,               // [num_tokens, num_heads * head_size]
    void* key,                 // [num_tokens, num_kv_heads * head_size]
    int head_size,
    const void* cos_sin_cache,  // [max_position, rot_dim]
    int num_tokens,
    int rot_dim,
    int num_heads,
    int num_kv_heads,
    int dtype);
} // namespace cuda