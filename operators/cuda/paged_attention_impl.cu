#include "paged_attention_impl.h"
#include <vector>

namespace cuda {

inline OrtStatusPtr CudaCall(cudaError_t cuda_error) {
  if (cuda_error == cudaSuccess) return nullptr;
  return OrtW::API::CreateStatus(ORT_FAIL, MakeString("cuda error:", (int)cuda_error).c_str());
}

void InitializeHeadMapping(void* dest_data, const void* src_data, size_t count) {
  cudaMemcpy(dest_data, src_data, count, cudaMemcpyHostToDevice);
}

template <typename T>
OrtStatusPtr CheckInputs(const cudaStream_t stream, OrtAllocator* allocator, const ortc::Tensor<T>& query, const ortc::Tensor<T>& key,
                         const ortc::Tensor<T>& value, const ortc::Tensor<T>& key_cache, const ortc::Tensor<T>& value_cache,
                         const ortc::Tensor<int32_t>& block_tables, const ortc::Tensor<int32_t>& slot_mappings, 
                         std::optional<const ortc::Tensor<int32_t>*> context_lens,
                         std::optional<const ortc::Tensor<int64_t>*> positions, InputMetadata& input_metadata) {
    const std::vector<int64_t>& query_shape = query.Shape();
    if (query_shape.size() < 2 || query_shape.size() > 3) {
      return OrtW::CreateStatus(MakeString("Invalid query shape, expect 2 or 3 dimensions"), ORT_INVALID_ARGUMENT);
    }
    if (query_shape.back() != num_heads_ * head_size_) {
      return OrtW::CreateStatus(MakesString("query shape should equal to num_heads_ * head_size_"));
    }

    // TODO(leca): Cpu input or CUDA input?
    int seq_len = query_shape.size() == 3 ? query_shape[1] : query_shape[0];
    if (positions.has_value()) {
      std::vector<int64_t> positions_host((*positions)->Shape().size());
      ORTX_RETURN_IF_ERROR(CudaCall(cudaMemcpy(positions_host.data(), (*positions)->DataRaw(), (*positions)->SizeInBytes(), cudaMemcpyDeviceToHost)));
      while (positions_host.back() == 0) {
        positions_host.pop_back();
        seq_len--;
      }

      input_metadata.max_num_blocks_per_seq = 0;
      // in prompt mode
      if (positions_host.size() > 1 || positions_host.back() == 0) {
        input_metadata.num_prompt_tokens = seq_len;
        input_metadata.num_generation_tokens = 0;
      } else {
        input_metadata.num_prompt_tokens = 0;
        input_metadata.num_generation_tokens = seq_len;
        input_metadata.max_context_len = positions_host.back() + 1; // TODO(leca): what if position_host is empty?

        int32_t block_size = gsl::narrow<int32_t>(key_cache.Shape()[3]);
        for (int i = 0; i < positions_host.back() + 1; i += block_size) input_metadata.max_num_blocks_per_seq++;
      }
    } else {
      // TODO(leca): context_lens is nullptr?
      std::vector<int32_t> context_len_host((*context_lens)->SizeInBytes());
      ORTX_RETURN_IF_ERROR(CudaCall(cudaMemcpy(context_len_host.data(), *(context_lens)->DataRaw(), *(context_lens)->SizeInBytes(), cudaMemcpyDeviceToHost)));
      std::vector<int64_t> position_ids;
      for (size_t i = 0; i < context_len_host.size(); i++) {
        if (context_len_host[i] == 0)   continue;
        std::vector<int64_t> position_id(context_len_host[i]);
        std::iota(position_id.begin(), position_id.end(), 0);   // fill position_id with {0, 1, 2, ...context_len_span[i]-1}
        position_ids.insert(position_ids.end(), position_id.begin(), position_id.end());
      }
      input_metadata.position_ids = GetScratchBuffer<int64_t>(allocator->Alloc(allocator, cnt), allocator);
      ORTX_RETURN_IF_ERROR(CudaCall(cudaMemcpyAsync(input_metadata.position_ids.get(), position_ids.data(), position_ids.size(), cudaMemcpyHostToDevice, stream)));
    }
    input_metadata.num_valid_tokens = seq_len;
  
  return nullptr;
}

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
    int dtype) {

}

template<typename T>
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
    int dtype) {

}

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
    int dtype) {

}

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
    int dtype) {

}

}   // namespace cuda