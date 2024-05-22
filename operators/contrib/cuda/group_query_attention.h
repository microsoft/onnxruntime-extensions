// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "cuda_type.h"
#include "ortx_common.h"
#include "../attention_common.h"
#include "group_query_attention_impl.cuh"
#include "device_prop.cuh"
#if USE_FLASH_ATTENTION
#include "flash_attention/flash_api.h"
#endif
#if USE_MEMORY_EFFICIENT_ATTENTION
#include "cutlass_fmha/memory_efficient_attention.h"
#endif

/*
 * Usage:
 * pip3 install . --config-settings "ortx-user-option=use-cuda,cc_debug,ort_pkg_dir=/home/leca/ort_pkg_19"
 * python3 test_cudaops.py TestCudaOps.test_cuda_GroupQueryAttention
 */

namespace contrib {

template <typename T>
using UniquePtrWithDeletor = std::unique_ptr<T, std::function<void(T*)>>;

template <typename T>
inline UniquePtrWithDeletor<T> GetScratchBuffer(void* p, OrtAllocator* allocator) {
  return UniquePtrWithDeletor<T>{static_cast<T*>(p), [allocator = std::move(allocator)](T* p) {
                                  allocator->Free(allocator, p);
                                }};
}

template <typename T>
OrtStatusPtr CheckInputs(const Ort::Custom::Tensor<T>& query,
                   std::optional<const ortc::Tensor<T>*> key,
                   std::optional<const ortc::Tensor<T>*> value,
                   std::optional<const ortc::Tensor<T>*> past_key,
                   std::optional<const ortc::Tensor<T>*> past_value,
                   std::optional<const ortc::Tensor<T>*> cos_cache,
                   std::optional<const ortc::Tensor<T>*> sin_cache,
                   void* parameters,
                   int num_heads,
                   int kv_num_heads,
                   const Ort::Custom::Tensor<int>& seqlens_k,
                   const Ort::Custom::Tensor<int>& total_seqlen,
                   bool is_past_bsnh,
                   float scale,
                   int max_threads_per_block) {
  if (max_threads_per_block > 0 && num_heads > max_threads_per_block) {
    return OrtW::CreateStatus(MakeString("num_heads should be no larger than ", max_threads_per_block), ORT_INVALID_ARGUMENT);
  }                

  // Note: Here S* is past_cache_sequence_length, S- is past_sequence_length, S+ is sequence_length
  //     past_key                   : (B, N_k, S*, H) or (B, N_k, S-, H) or nullptr
  //     past_value                 : (B, N_k, S*, H) or (B, N_k, S-, H) or nullptr
  // no packing for q/k/v:
  //     query            (Q)       : (B, S, D) or (B, S, (D_q + 2 D_kv))
  //     key              (K)       : (B, S, D_kv) or nullptr
  //     value            (V)       : (B, S, D_kv) or nullptr

  AttentionQkvFormat qkv_format = Q_K_V_BSNH;
  AttentionQkvFormat past_kv_format = is_past_bsnh ? Q_K_V_BSNH : Q_K_V_BNSH;
  const bool is_packed_qkv = !key.has_value();
  const auto& query_dims = query.Shape();

  if (query_dims.size() != 3) {
    return OrtW::CreateStatus(MakeString("Input 'query' is expected to have 3 dimensions, got ", query_dims.size()), ORT_INVALID_ARGUMENT);
  }

  int batch_size = static_cast<int>(query_dims[0]);
  int sequence_length = static_cast<int>(query_dims[1]);
  int q_hidden_size = static_cast<int>(query_dims[2]);
  int head_size = 0;

  if (num_heads % kv_num_heads != 0) {
    return OrtW::CreateStatus(MakeString("num_heads must be a multiple of kv_num_heads. Got num_heads % kv_num_heads == ", num_heads % kv_num_heads), ORT_INVALID_ARGUMENT);
  }

  int kv_hidden_size = 0;
  // Check key and value when not packed
  if (!is_packed_qkv) {
    head_size = static_cast<int>(q_hidden_size) / num_heads;
    if (head_size % 8 != 0) {
      return OrtW::CreateStatus(MakeString("head_size must be a multiple of 8. Got head_size % 8 == ", head_size % 8), ORT_INVALID_ARGUMENT);
    }
    if (!value.has_value()) {
      return OrtW::CreateStatus("Input 'key' and 'value' shall be both present, or both absent in the case of packed qkv.", ORT_INVALID_ARGUMENT);
    }
    const auto& key_dims = (*key)->Shape();
    if (key_dims.size() != 3) {
      return OrtW::CreateStatus(MakeString("Input 'key' is expected to have 3 dimensions, got ", key_dims.size()), ORT_INVALID_ARGUMENT);
    } else if (query_dims[0] != key_dims[0]) {
      return OrtW::CreateStatus("Input 'query' and 'key' shall have same dim 0 (batch size)", ORT_INVALID_ARGUMENT);
    } else if (query_dims[1] != key_dims[1]) {
      return OrtW::CreateStatus("Input 'query' and 'key' shall have same dim 1 (sequence length)", ORT_INVALID_ARGUMENT);
    }
    kv_hidden_size = static_cast<int>(key_dims[2]);
    const auto& value_dims = (*value)->Shape();
    if (value_dims.size() != 3) {
      return OrtW::CreateStatus(MakeString("Input 'value' is expected to have 3 dimensions, got ", value_dims.size()), ORT_INVALID_ARGUMENT);
    } else if (query_dims[0] != value_dims[0]) {
      return OrtW::CreateStatus("Input 'query' and 'value' shall have same dim 0 (batch size)", ORT_INVALID_ARGUMENT);
    } else if (query_dims[1] != value_dims[1]) {
      return OrtW::CreateStatus("Input 'query' and 'value' shall have same dim 1 (sequence length)", ORT_INVALID_ARGUMENT);
    } else if (value_dims[2] != kv_hidden_size) {
      return OrtW::CreateStatus("Input 'value' is expected to have same hidden size as key.", ORT_INVALID_ARGUMENT);
    }
  } else {
    // Check packed qkv
    head_size = static_cast<int>(q_hidden_size) / (num_heads + 2 * kv_num_heads);
    if (head_size % 8 != 0) {
      return OrtW::CreateStatus(MakeString("head_size must be a multiple of 8. Got head_size % 8 == ", head_size % 8), ORT_INVALID_ARGUMENT);
    }
    if (value.has_value()) {
      return OrtW::CreateStatus("Input 'key' and 'value' shall be both present, or both absent in the case of packed qkv.", ORT_INVALID_ARGUMENT);
    }
    q_hidden_size = head_size * num_heads;
    kv_hidden_size = head_size * kv_num_heads;
  }

  // Check past-present KV
  int32_t past_sequence_length = 0;
  if (past_key.has_value() && past_value.has_value()) {
    const auto& past_key_dims = (*past_key)->Shape();
    const auto& past_value_dims = (*past_value)->Shape();

    if (past_key_dims.size() != 4) {
      return OrtW::CreateStatus(MakeString("Input 'past_key' is expected to have 4 dimensions, got ", past_key_dims.size()), ORT_INVALID_ARGUMENT);
    }
    if (past_value_dims.size() != 4) {
      return OrtW::CreateStatus(MakeString("Input 'past_value' is expected to have 4 dimensions, got ", past_value_dims.size()), ORT_INVALID_ARGUMENT);
    }

    if (past_key_dims[0] != batch_size) {
      return OrtW::CreateStatus(MakeString("Input 'past_key' dimension 0 should be batch_size, got ", past_key_dims[0]), ORT_INVALID_ARGUMENT);
    }
    if (past_value_dims[0] != batch_size) {
      return OrtW::CreateStatus(MakeString("Input 'past_value' dimension 0 should be batch_size, got ", past_value_dims[0]), ORT_INVALID_ARGUMENT);
    }

    // BNSH
    if (!is_past_bsnh) {
      if (past_key_dims[2] != past_value_dims[2]) {
        return OrtW::CreateStatus(MakeString("BNSH Input 'past_key' and 'past_value' should have same dimension 2 (max sequence length or past sequence length), got ", past_key_dims[1]), ORT_INVALID_ARGUMENT);
      }
      if (past_key_dims[1] != kv_num_heads) {
        return OrtW::CreateStatus("Input 'past_key' shall have kv_num_heads", ORT_INVALID_ARGUMENT);
      }
      if (past_value_dims[1] != kv_num_heads) {
        return OrtW::CreateStatus("Input 'past_value' shall have kv_num_heads", ORT_INVALID_ARGUMENT);
      }
      // We assume all sequence in past kv are right-padded to max or past sequence length
      past_sequence_length = static_cast<int>(past_key_dims[2]);
      // BSNH
    } else {
      if (past_key_dims[1] != past_value_dims[1]) {
        return OrtW::CreateStatus(MakeString("BNSH Input 'past_key' and 'past_value' should have same dimension 1 (max sequence length or past sequence length), got ", past_key_dims[1]), ORT_INVALID_ARGUMENT);
      }
      if (past_key_dims[2] != kv_num_heads) {
        return OrtW::CreateStatus("Input 'past_key' shall have kv_num_heads", ORT_INVALID_ARGUMENT);
      }
      if (past_value_dims[2] != kv_num_heads) {
        return OrtW::CreateStatus("Input 'past_value' shall have kv_num_heads", ORT_INVALID_ARGUMENT);
      }
      // We assume all sequence in past kv are right-padded to max or past sequence length
      past_sequence_length = static_cast<int>(past_key_dims[1]);
    }

    if (past_key_dims[3] != head_size) {
      return OrtW::CreateStatus(MakeString("Input 'past_key' dimension 3 should be same as head_size, got ", past_key_dims[3]), ORT_INVALID_ARGUMENT);
    }
    if (past_value_dims[3] != head_size) {
      return OrtW::CreateStatus(MakeString("Input 'past_value' dimension 3 should be same as head_size, got ", past_value_dims[3]), ORT_INVALID_ARGUMENT);
    }
  } else if (past_key.has_value() || past_value.has_value()) {
    return OrtW::CreateStatus("Input 'past_key' and 'past_value' shall be both present or both absent.", ORT_INVALID_ARGUMENT);
  }

  // Check seqlens_k tensor (holding past seqlen for token gen)
  const auto& seqlens_dim = seqlens_k.Shape();
  if (seqlens_dim.size() != 1 && seqlens_dim[0] != batch_size) {
    return OrtW::CreateStatus("seqlens_k must be shape (batch_size).", ORT_INVALID_ARGUMENT);
  }

  // Set present sequence length and kv_share_buffer from input total_seqlen tensor
  size_t num_dimensions = total_seqlen.Shape().size();
  int64_t shape_size = total_seqlen.NumberOfElement();
  if (!IsScalarOr1ElementVector(num_dimensions, shape_size)) {
    return OrtW::CreateStatus("total_sequence_length tensor must be of one element.", ORT_INVALID_ARGUMENT);
  }
  int total_sequence_length = *(total_seqlen.Data());
  int present_sequence_length = std::max(total_sequence_length, past_sequence_length);

  if (cos_cache.has_value() && sin_cache.has_value()) {
    const auto& cos_dims = (*cos_cache)->Shape();
    const auto& sin_dims = (*sin_cache)->Shape();

    if (head_size % 16 != 0) {
      return OrtW::CreateStatus(MakeString("head_size shall be a multiple of 16. Got head_size % 16 == ", head_size % 16), ORT_INVALID_ARGUMENT);
    }
    if (cos_dims[0] != present_sequence_length) {
      return OrtW::CreateStatus("cos_cache dimension 0 must be of present_sequence_length.", ORT_INVALID_ARGUMENT);
    }
    if (sin_dims[0] != present_sequence_length) {
      return OrtW::CreateStatus("sin_cache dimension 0 must be of present_sequence_length.", ORT_INVALID_ARGUMENT);
    }
    if (cos_dims[1] != (head_size / 16) * 8) {
      return OrtW::CreateStatus("cos_cache dimension 1 must be <= head_size / 2 and a multiple of 8.", ORT_INVALID_ARGUMENT);
    }
    if (sin_dims[1] != (head_size / 16) * 8) {
      return OrtW::CreateStatus("sin_cache dimension 1 must be <= head_size / 2 and a multiple of 8.", ORT_INVALID_ARGUMENT);
    }
  } else if (cos_cache.has_value() || sin_cache.has_value()) {
    return OrtW::CreateStatus("Input 'cos_cache' and 'sin_cache' shall be both present or both absent.", ORT_INVALID_ARGUMENT);
  }

  bool is_prompt = sequence_length != 1;

  if (parameters != nullptr) {
    GroupQueryAttentionParameters* output_parameters = reinterpret_cast<GroupQueryAttentionParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->sequence_length = sequence_length;                  // sequence length of Q
    output_parameters->seqlen_past_kv_cache = past_sequence_length;        // max sequence length of past kv tensors
    output_parameters->seqlen_present_kv_cache = present_sequence_length;  // max sequence length of present kv tensors
    output_parameters->hidden_size = q_hidden_size;
    output_parameters->num_heads = num_heads;
    output_parameters->head_size = head_size;
    output_parameters->kv_hidden_size = kv_hidden_size;
    output_parameters->kv_num_heads = kv_num_heads;
    output_parameters->is_packed_qkv = is_packed_qkv;
    output_parameters->is_unidirectional = true;
    output_parameters->is_prompt = is_prompt;
    output_parameters->scale = scale;
    output_parameters->qkv_format = qkv_format;
    output_parameters->past_kv_format = past_kv_format;
  }

  return nullptr;
}

template <typename T>
struct GroupQueryAttention {
  static OrtMemType GetInputMemoryType(size_t input_index) {
//    if (input_index == 6) return OrtMemType::OrtMemTypeCPUInput;  // total_seqlen
    if (input_index == 4) return OrtMemType::OrtMemTypeCPUInput;  // total_seqlen
    return OrtMemType::OrtMemTypeDefault;
  }

  static size_t GetMayInplace(int** input_index, int** output_index) {  // past_key <=> key, past_value <=> value
    size_t ret = 2;
    *input_index = static_cast<int*>(malloc(ret * sizeof(int)));
//    (*input_index)[0] = 3;
//    (*input_index)[1] = 4;
    (*input_index)[0] = 5;
    (*input_index)[1] = 6;
    *output_index = static_cast<int*>(malloc(ret * sizeof(int)));
    (*output_index)[0] = 1;
    (*output_index)[1] = 2;
    return 2;
  }

  static void ReleaseMayInplace(int* input_index, int* output_index) {
    free(input_index);
    free(output_index);
  }

  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    int64_t num_heads = 0, kv_num_heads = 0;
    ORTX_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "num_heads", num_heads));
    ORTX_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "kv_num_heads", kv_num_heads));
    num_heads_ = static_cast<int>(num_heads);
    kv_num_heads_ = static_cast<int>(kv_num_heads);
    is_past_bsnh_ = false;
    local_window_size_ = static_cast<int>(OrtW::GetOpAttributeOrDefault<int64_t>(info, "local_window_size", -1));
    do_rotary_ = OrtW::GetOpAttributeOrDefault<int64_t>(info, "do_rotary", 0) == 1;
    rotary_interleaved_ = OrtW::GetOpAttributeOrDefault<int64_t>(info, "rotary_interleaved", 0) == 1;
    scale_ = OrtW::GetOpAttributeOrDefault(info, "scale", 0.0f);
    
#if USE_FLASH_ATTENTION
    disable_flash_attention_ = sizeof(T) != 2 || ParseEnvironmentVariableWithDefault<bool>(attention::kDisableFlashAttention, false);
#else
    disable_flash_attention_ = true;
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
    disable_memory_efficient_attention_ = sizeof(T) != 2 ||
                                          ParseEnvironmentVariableWithDefault<bool>(attention::kDisableMemoryEfficientAttention, false);
#else
    disable_memory_efficient_attention_ = true;
#endif

    if (!disable_flash_attention_) {
      OrtAllocator* allocator = nullptr;
      ORTX_RETURN_IF_ERROR(api.KernelInfoGetAllocator(&info, OrtMemType::OrtMemTypeDefault, &allocator));
      allocator_ = UniquePtrWithDeletor<OrtAllocator>{allocator, [&api](OrtAllocator* p){api.ReleaseAllocator(p);}};
      zeros_ = GetScratchBuffer<int>(allocator_->Alloc(allocator_.get(), kZerosCount), allocator_.get());
    }
    return nullptr;
  }

  OrtStatusPtr Compute(OrtKernelContext* kernel_context, const Ort::Custom::CudaContext& ctx, const ortc::Tensor<T>& query, std::optional<const ortc::Tensor<T>*> key,
//                       std::optional<const ortc::Tensor<T>*> value, std::optional<const ortc::Tensor<T>*> past_key, std::optional<const ortc::Tensor<T>*> past_value,
//                       const ortc::Tensor<int>& seqlens_k, const ortc::Tensor<int>& total_seqlen, std::optional<const ortc::Tensor<T>*> cos_cache, 
                       std::optional<const ortc::Tensor<T>*> value, const ortc::Tensor<int>& seqlens_k, const ortc::Tensor<int>& total_seqlen,
                       std::optional<const ortc::Tensor<T>*> past_key, std::optional<const ortc::Tensor<T>*> past_value, std::optional<const ortc::Tensor<T>*> cos_cache, 
                       std::optional<const ortc::Tensor<T>*> sin_cache, ortc::Tensor<T>& attn_out, std::optional<ortc::Tensor<T>*> present_key, std::optional<ortc::Tensor<T>*> present_value) const {
    GroupQueryAttentionParameters parameters;
    ORTX_RETURN_IF_ERROR(CheckInputs<T>(query, key, value, past_key, past_value, cos_cache, sin_cache, &parameters, num_heads_, kv_num_heads_, 
                                     seqlens_k, total_seqlen, is_past_bsnh_, scale_, DeviceProp::GetCudaDeviceProp().maxThreadsPerBlock));
    parameters.local_window_size = local_window_size_;
    parameters.is_unidirectional = is_unidirectional_;
    parameters.zeros_count = kZerosCount;
    parameters.zero_ptr = zeros_.get();
    // parameters.left_padding = left_padding_;
    int sequence_length = parameters.sequence_length;
    parameters.do_rotary = do_rotary_;
    parameters.rotary_interleaved = rotary_interleaved_;
  
    std::vector<int64_t> output_shape(3, 0);
    output_shape[0] = static_cast<int64_t>(parameters.batch_size);
    output_shape[1] = static_cast<int64_t>(sequence_length);
    output_shape[2] = static_cast<int64_t>(parameters.hidden_size);

    OrtMemoryInfo* mem_info = nullptr;
    ORTX_RETURN_IF_ERROR(OrtW::API::CreateOrtMemoryInfo("Cuda", OrtDeviceAllocator, ctx.device_id, OrtMemTypeDefault, &mem_info));
  
#if USE_FLASH_ATTENTION
    bool use_flash_attention = !disable_flash_attention_ && flash::is_supported(DeviceProp::GetCudaDeviceProp(), parameters.head_size, parameters.num_heads, parameters.kv_num_heads);
    // Allocate buffers
    size_t softmax_lse_bytes = 0;
    size_t softmax_lse_accum_bytes = 0;
    size_t out_accum_bytes = 0;
    if (use_flash_attention) {
      // softmax buffer
      softmax_lse_bytes = flash::get_softmax_lse_size(parameters.sequence_length, parameters.batch_size, parameters.num_heads);
      // split kv buffer
      using namespace std;
      auto [num_splits, slse_accum_bytes, o_accum_bytes] = flash::get_num_splits_and_buffer_sizes(
          parameters.batch_size, parameters.sequence_length, parameters.sequence_length, parameters.num_heads,
          parameters.head_size, DeviceProp::GetCudaDeviceProp().multiProcessorCount);
      parameters.num_splits = num_splits;
      softmax_lse_accum_bytes = slse_accum_bytes;
      out_accum_bytes = o_accum_bytes;
    }
    void* softmax_lse_p = nullptr;
    ORTX_RETURN_IF_ERROR(OrtW::API::KernelContextGetScratchBuffer(kernel_context, mem_info, softmax_lse_bytes, &softmax_lse_p));
    auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_p, allocator_.get());

    void* softmax_lse_accum_p = nullptr;
    ORTX_RETURN_IF_ERROR(OrtW::API::KernelContextGetScratchBuffer(kernel_context, mem_info, softmax_lse_accum_bytes, &softmax_lse_accum_p));
    auto softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_p, allocator_.get());

    void* out_accum_p = nullptr;
    ORTX_RETURN_IF_ERROR(OrtW::API::KernelContextGetScratchBuffer(kernel_context, mem_info, out_accum_bytes, &out_accum_p));
    auto out_accum_buffer = GetScratchBuffer<void>(out_accum_p, allocator_.get());
#else
    constexpr bool use_flash_attention = false;
    UniquePtrWithDeletor<void> softmax_lse_buffer = nullptr;
    UniquePtrWithDeletor<void> softmax_lse_accum_buffer = nullptr;
    UniquePtrWithDeletor<void> out_accum_buffer = nullptr;
#endif
  
#if USE_MEMORY_EFFICIENT_ATTENTION
    int sm = (DeviceProp::GetCudaDeviceProp().major * 10) + DeviceProp::GetCudaDeviceProp().minor;
    bool use_memory_efficient_attention =
        !use_flash_attention &&
        !disable_memory_efficient_attention_ &&
        local_window_size_ == -1 &&
        do_rotary_ == false &&
        key != nullptr &&
        (parameters.head_size & 7) == 0 &&
        parameters.sequence_length <= parameters.seqlen_past_kv_cache + parameters.sequence_length &&
        (sizeof(T) == 2 || parameters.sequence_length >= attention::kMinSeqLenForMemoryEfficientAttentionFp32) &&
        cuda::has_memory_efficient_attention(sm, sizeof(T) == 2);
    // allocate buffers
    size_t kv_buffer_bytes = 0;
    // need a buffer if we must ungroup kv
    const bool needs_buff = (parameters.num_heads != parameters.kv_num_heads);
    if (use_memory_efficient_attention && needs_buff) {
      kv_buffer_bytes = (sizeof(T) * parameters.batch_size * parameters.num_heads * parameters.seqlen_present_kv_cache * parameters.head_size);
    }
    size_t fmha_buffer_bytes = 0;
    if (use_memory_efficient_attention && cuda::MemoryEfficientAttentionParams::need_workspace(parameters.head_size, sizeof(T) == sizeof(float))) {
      fmha_buffer_bytes = (parameters.batch_size * parameters.sequence_length * parameters.num_heads * parameters.head_size * sizeof(float));
    }
    void* k_p = nullptr;
    ORTX_RETURN_IF_ERROR(OrtW::API::KernelContextGetScratchBuffer(kernel_context, mem_info, kv_buffer_bytes, &k_p));
    auto k_buffer = GetScratchBuffer<void>(k_p, allocator_.get());

    void* v_p = nullptr;
    ORTX_RETURN_IF_ERROR(OrtW::API::KernelContextGetScratchBuffer(kernel_context, mem_info, kv_buffer_bytes, &v_p));
    auto v_buffer = GetScratchBuffer<void>(v_p, allocator_.get());

    void* fmha_p = nullptr;
    ORTX_RETURN_IF_ERROR(OrtW::API::KernelContextGetScratchBuffer(kernel_context, mem_info, fmha_buffer_bytes, &fmha_p));
    auto fmha_buffer = GetScratchBuffer<void>(fmha_p, allocator_.get());
#else
    constexpr bool use_memory_efficient_attention = false;
    UniquePtrWithDeletor<void> k_buffer = nullptr;
    UniquePtrWithDeletor<void> v_buffer = nullptr;
    UniquePtrWithDeletor<void> fmha_buffer = nullptr;
#endif
  
    // seqlens_k buffer
    size_t seqlens_k_bytes = sizeof(int) * parameters.batch_size;
    void* seqlens_p = nullptr;
    ORTX_RETURN_IF_ERROR(OrtW::API::KernelContextGetScratchBuffer(kernel_context, mem_info, seqlens_k_bytes, &seqlens_p));
    auto seqlens_k_buffer = GetScratchBuffer<void>(seqlens_p, allocator_.get());

    std::vector<int64_t> present_dims;
    if (parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BSNH) {
      present_dims = {
          parameters.batch_size, parameters.seqlen_present_kv_cache, parameters.kv_num_heads, parameters.head_size};
    } else {  // BNSH
      present_dims = {
          parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache, parameters.head_size};
    }
  
    using TT = typename CudaT<T>::MappedType;
    cuda::GroupQueryAttentionData<TT> data;
    data.query = reinterpret_cast<const TT*>(query.Data());
    data.key = key.has_value() ? reinterpret_cast<const TT*>((*key)->Data()) : nullptr;
    data.value = value.has_value() ? reinterpret_cast<const TT*>((*value)->Data()) : nullptr;
    data.past_key = past_key.has_value() ? reinterpret_cast<const TT*>((*past_key)->Data()) : nullptr;
    data.past_value = past_value.has_value() ? reinterpret_cast<const TT*>((*past_value)->Data()) : nullptr;
    data.output = reinterpret_cast<TT*>(attn_out.Allocate(output_shape));
    data.present_key = present_key.has_value() ? reinterpret_cast<TT*>((*present_key)->Allocate(present_dims)) : nullptr;
    data.present_value = present_value.has_value() ? reinterpret_cast<TT*>((*present_value)->Allocate(present_dims)) : nullptr;
    data.seqlens_k = const_cast<int*>(seqlens_k.Data());
    data.use_flash_attention = use_flash_attention;
    data.use_memory_efficient_attention = use_memory_efficient_attention;
    if (data.past_key == data.present_key) {
      parameters.kv_share_buffer = true;
    } else {
      parameters.kv_share_buffer = false;
    }
    // Flash Buffers
    if (softmax_lse_buffer != nullptr) {
      data.softmax_lse = reinterpret_cast<TT*>(softmax_lse_buffer.get());
    }
    if (softmax_lse_accum_buffer != nullptr) {
      data.softmax_lse_accum = reinterpret_cast<TT*>(softmax_lse_accum_buffer.get());
    }
    if (out_accum_buffer != nullptr) {
      data.out_accum = reinterpret_cast<TT*>(out_accum_buffer.get());
    }
    if (seqlens_k_buffer != nullptr) {
      data.seqlens_k_total = reinterpret_cast<int*>(seqlens_k_buffer.get());
    }
    // Memory Efficient Buffers
    if (k_buffer != nullptr) {
      data.k = reinterpret_cast<TT*>(k_buffer.get());
      data.v = reinterpret_cast<TT*>(v_buffer.get());
    }
    if (fmha_buffer != nullptr) {
      data.fmha_buffer = reinterpret_cast<TT*>(fmha_buffer.get());
    }
    if (k_buffer != nullptr) {
      data.k = reinterpret_cast<TT*>(k_buffer.get());
      data.v = reinterpret_cast<TT*>(v_buffer.get());
    }
    if (k_buffer != nullptr) {
      data.k = reinterpret_cast<TT*>(k_buffer.get());
      data.v = reinterpret_cast<TT*>(v_buffer.get());
    }
    if (fmha_buffer != nullptr) {
      data.fmha_buffer = reinterpret_cast<TT*>(fmha_buffer.get());
    }
    // Rotary
    if (parameters.do_rotary) {
      data.cos_cache = reinterpret_cast<const TT*>((*cos_cache)->Data());
      data.sin_cache = reinterpret_cast<const TT*>((*sin_cache)->Data());
    }

    OrtW::API::ReleaseMemoryInfo(mem_info);
    return cuda::QkvToContext<TT>(
        /*device_prop, ctx.cublas,*/ reinterpret_cast<cudaStream_t>(ctx.cuda_stream), parameters, data);
  }

 private:
  int num_heads_;     // number of attention heads
  int kv_num_heads_;  // different for k and v for group query attention
  int local_window_size_;
  bool is_unidirectional_;
  bool is_past_bsnh_;
  bool do_rotary_;
  bool rotary_interleaved_;
  float scale_;
  bool disable_flash_attention_;
  bool disable_memory_efficient_attention_;
  static constexpr int kZerosCount = 256;  // In prompt case we create a zero buffer of size 256 for seqlen (assume batch_size <= 256)
  UniquePtrWithDeletor<OrtAllocator> allocator_; // TODO(leca): Does the release order of allocator_ and zeros_ matter?
  UniquePtrWithDeletor<int> zeros_;
};

}  // namespace contrib