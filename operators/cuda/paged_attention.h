// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "paged_attention_impl.h"

template<typename T>
struct PagedAttention {
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    int64_t num_heads = 0, head_size = 0;
    ORTX_RETURN_IF_ERROR(api.KernelInfoGetAttribute_int64(&info, "num_heads", &num_heads));
    assert(num_heads > 0);
    num_heads_ = static_cast<int32_t>(num_heads);
    num_kv_heads_ = static_cast<int32_t>(OrtW::GetOpAttributeOrDefault<int64_t>(info, "num_kv_heads", num_heads));
    
    ORTX_RETURN_IF_ERROR(api.KernelInfoGetAttribute_int64(&info, "head_size", &head_size));
    assert(head_size > 0);
    head_size_ = static_cast<int32_t>(head_size);

    ORTX_RETURN_IF_ERROR(api.KernelInfoGetAttribute_float(&info, "scale", &scale_));
    assert(scale_ > 0);

    num_queries_per_kv_ = num_heads_ / num_kv_heads_;
    std::vector<int32_t> head_mapping_host(num_heads_);
    for (int i = 0; i < num_kv_heads_; i++) {
      for (int j = 0; j < num_queries_per_kv_; j++) {
        head_mapping_host[i * num_queries_per_kv_ + j] = i;
      }
    }

    OrtAllocator* allocator = nullptr;
    ORTX_RETURN_IF_ERROR(api.KernelInfoGetAllocator(&info, OrtMemType::OrtMemTypeDefault, &allocator));
    allocator_ = UniquePtrWithDeletor<OrtAllocator>{allocator, [&api](OrtAllocator* p){api.ReleaseAllocator(p);}};
    head_mapping_ = GetScratchBuffer<int32_t>(allocator_->Alloc(allocator_.get(), num_heads_), allocator_.get());
    InitializeHeadMapping(head_mapping_.get(), head_mapping_host.data(), head_mapping_host.size());
  }

  OrtStatusPtr Compute(Ort::Custom::CUDAKernelContext* ctx, const ortc::Tensor<T>& query, const ortc::Tensor<T>& key,
                       const ortc::Tensor<T>& value, const ortc::Tensor<T>& key_cache, const ortc::Tensor<T>& value_cache,
                       const ortc::Tensor<int32_t>& block_tables, const ortc::Tensor<int32_t>& slot_mappings, 
                       std::optional<const ortc::Tensor<int32_t>*> context_lens,
                       std::optional<const ortc::Tensor<int64_t>*> positions
                       std::optional<const ortc::Tensor<T>*> cos_sin_cache, ortc::Tensor<T>& attn_out) const {
    InputMetadata input_metadata;
    ORTX_RETURN_IF_ERROR(CheckInputs(ctx.GetCudaStream(), allocator_.get(), query, key, value, key_cache, value_cache, block_tables, slot_mappings, context_lens, positions, input_metadata));
    const std::vector<int64_t>& query_shape = query.Shape();
    T* output_data = attn_out.Allocate(query_shape);

    if (cos_sin_cache.has_value()) {
      int64_t rot_dim = (*cos_sin_cache)->Shape()[1];
      assert(rot_dim == head_size_);
      rotary_embedding_neox(reinterpret_cast<cudaStream_t>(ctx.GetCudaStream()), (*positions)->Data<int64_t>(), query.DataRaw(), key.DataRaw(), head_size_,
                            (*cos_sin_cache)->DataRaw(), input_metadata.num_valid_tokens, rot_dim, num_heads_, num_kv_heads_, 1);
    }

    const std::vector<int64_t>& key_cache_shape = key_cache.Shape();
    if (input_metadata.num_valid_tokens > 0 && key_cache_shape.size() > 3) {
      int64_t key_shape_r[3] = {input_metadata.num_valid_tokens, num_kv_heads_, head_size_};
      int64_t value_shape_r[3] = {input_metadata.num_valid_tokens, num_kv_heads_, head_size_};
      int block_size = gsl::narrow<int>(key_cache_shape[3]);
      reshape_and_cache(reinterpret_cast<cudaStream_t>(ctx.GetCudaStream()), key.DataRaw(), value.DataRaw(), key_cache.DataRaw(), value_cache.DataRaw(), slot_mappings.Data(),
                        key_shape_r, value_shape_r, block_size, key_cache_shape[4], 1);
    }

    using TT = typename CudaT<T>::MappedType;
    if (input_metadata.num_prompt_tokens > 0) {
      //TODO(leca): flash attention for prompt > 0 case
      return nullptr; // Don't handle prompt with decoding case for now
    }

    if (input_metadata.num_generation_tokens > 0) {
      constexpr int PARTITION_SIZE = 512;
      int max_num_partitions = (input_metadata.max_context_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
      bool use_v1 = max_num_partitions == 1 || (query_shape[0] * query_shape[1]) > PARTITION_SIZE;
      int64_t generation_qeury_shape[3] = {input_metadata.num_valid_tokens, num_heads_, head_size_};
      if (use_v1) {
        paged_attention_v1(reinterpret_cast<cudaStream_t>(ctx.GetCudaStream()), reinterpret_cast<TT*>(output_data), query.DataRaw(),
                           key_cache.DataRaw(), value_cache.DataRaw(), head_mapping_.get(), scale_, 
                           block_tables.Data(), context_lens.has_value() ? (*context_lens)->Data() : nullptr,
                           value_cache.Shape()[3], input_metadata.max_context_len, nullptr,
                           input_metadata.max_num_blocks_per_seq, generation_qeury_shape, num_queries_per_kv_, 1);
      } else {
        OrtMemoryInfo* mem_info = nullptr;
        ORTX_RETURN_IF_ERROR(OrtW::API::CreateOrtMemoryInfo("Cuda", OrtDeviceAllocator, ctx.device_id, OrtMemTypeDefault, &mem_info));
        void* tmp_output_raw = ctx->GetScratchBufferUnderMultiStream(mem_info, query_shape.size() * max_num_partitions * sizeof(T));
        UniquePtrWithDeletor<T> tmp_output = GetScratchBuffer<T>(tmp_output_raw, allocator_.get());   // TODO(leca): should deallocate inside ORT
        void* exp_sums_raw = ctx->GetScratchBufferUnderMultiStream(mem_info, query_shape[0] * query_shape[1] * num_heads_ * max_num_partitions * sizeof(T));
        UniquePtrWithDeletor<T> exp_sums = GetScratchBuffer<T>(exp_sums_raw, allocator_.get());
        void* max_logits_raw = ctx->GetScratchBufferUnderMultiStream(mem_info, query_shape[0] * query_shape[1] * num_heads_ * max_num_partitions * sizeof(T));
        UniquePtrWithDeletor<T> max_logits = GetScratchBuffer<T>(max_logits_raw, allocator_.get());
        paged_attention_v2(reinterpret_cast<cudaStream_t>(ctx.GetCudaStream()), exp_sums_raw, max_logits_raw, tmp_output_raw, reinterpret_cast<TT*>(output_data), query.DataRaw(),
                           key_cache.DataRaw(), value_cache.DataRaw(), head_mapping_.get(), scale_,
                           block_tables.Data(), context_lens.has_value() ? (*context_lens)->Data() : nullptr,
                           value_cache.Shape()[3], input_metadata.max_context_len, nullptr,
                           input_metadata.max_num_blocks_per_seq, generation_qeury_shape, num_queries_per_kv_, 1);

        OrtW::API::ReleaseMemoryInfo(mem_info);
      }
    }
    return nullptr;
  }

private:
  int32_t num_heads_;                  // number of attention heads
  int32_t num_kv_heads_;                  // number of attention kv_heads
  int32_t head_size_;                      // number of attention heads
  float scale_;                            // sqrt(head_size_)
  UniquePtrWithDeletor<int32_t> head_mapping_;
  int32_t num_queries_per_kv_;
  UniquePtrWithDeletor<OrtAllocator> allocator_;
};