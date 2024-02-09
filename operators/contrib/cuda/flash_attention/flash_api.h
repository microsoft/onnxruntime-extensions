#pragma once

#if USE_FLASH_ATTENTION

#include <tuple>

namespace flash {

OrtStatusPtr mha_fwd(const cudaDeviceProp& dprops,
               cudaStream_t stream,
               void* q,            // batch_size x seqlen_q x num_heads x head_size
               void* k,            // batch_size x seqlen_k x num_heads_k x head_size
               void* v,            // batch_size x seqlen_k x num_heads_k x head_size
               void* out,          // batch_size x seqlen_q x num_heads x head_size
               void* softmax_lse,  // batch_size x num_heads x seqlen_q
               int batch_size,
               int num_heads,
               int num_heads_k,
               int head_size,
               int seqlen_q,
               int seqlen_k,
               float softmax_scale,
               bool is_causal,
               bool is_bf16,
               int num_splits = 0,
               void* softmax_lse_accum = nullptr,  // num_splits x batch_size x seqlen_q x num_heads
               void* out_accum = nullptr,          // num_splits x batch_size x seqlen_q x num_heads x head_size_rounded
               bool kv_bsnh = true,
               int local_window_size = -1);

OrtStatusPtr mha_varlen_fwd(const cudaDeviceProp& dprops,
                      cudaStream_t stream,
                      void* q,            // half (total_q, num_heads, head_size)
                      void* k,            // half (total_k, num_heads, head_size)
                      void* v,            // half (total_k, num_heads, v_head_size)
                      void* out,          // half (total_q, num_heads, v_head_size)
                      int* cu_seqlens_q,  // int (batch_size + 1)
                      int* cu_seqlens_k,  // int (batch_size + 1)
                      void* softmax_lse,  // float (batch_size, num_heads, max_seqlen_q)
                      int batch_size,
                      int num_heads,
                      int num_heads_k,
                      int head_size,
                      int max_seqlen_q,
                      int max_seqlen_k,
                      float softmax_scale,
                      bool is_causal,
                      bool is_bf16);

OrtStatusPtr mha_fwd_kvcache(const cudaDeviceProp& dprops,
                       cudaStream_t stream,
                       void* q,            // batch_size x seqlen_q x num_heads x head_size
                       void* kcache,       // batch_size x seqlen_k x num_heads_k x head_size or batch_size x num_heads_k seqlen_k x x head_size
                       void* vcache,       // batch_size x seqlen_k x num_heads_k x head_size or batch_size x num_heads_k seqlen_k x x head_size
                       void* k,            // batch_size x seqlen_k_new x num_heads_k x head_size
                       void* v,            // batch_size x seqlen_k_new x num_heads_k x head_size
                       void* out,          // batch_size x seqlen_q x num_heads x head_size
                       void* softmax_lse,  // batch_size x num_heads x seqlen_q
                       void* seqlens_k_,   // batch_size
                       int batch_size,
                       int num_heads,
                       int num_heads_k,
                       int head_size,
                       int seqlen_q,
                       int seqlen_k,
                       int seqlen_k_new,
                       const float softmax_scale,
                       bool is_causal,
                       bool is_bf16,
                       bool past_bsnh,  // otherwise bnsh
                       int num_splits = 0,
                       void* softmax_lse_accum = nullptr,  // num_splits x batch_size x seqlen_q x num_heads
                       void* out_accum = nullptr,          // num_splits x batch_size x seqlen_q x num_heads x head_size_rounded
                       int local_window_size = -1);

size_t get_softmax_lse_size(int max_seqlen_q, int batch_size, int num_heads);

std::tuple<int, int, int> get_num_splits_and_buffer_sizes(int batch_size, int seqlen_q, int seqlen_k, int num_heads,
                                                          int head_size, int num_SMs);

bool is_supported(const cudaDeviceProp& dprops, int head_size, int num_heads, int num_heads_k);

}  // namespace flash

#endif  //  USE_FLASH_ATTENTION
