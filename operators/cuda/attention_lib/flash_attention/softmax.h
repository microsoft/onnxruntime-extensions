/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void thread_reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& summary, Operator& op) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); mi++) {
    summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
#pragma unroll
    for (int ni = 1; ni < size<1>(tensor); ni++) {
      summary(mi) = op(summary(mi), tensor(mi, ni));
    }
  }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void quad_allreduce_(Tensor<Engine0, Layout0>& dst, Tensor<Engine1, Layout1>& src, Operator& op) {
  CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
  for (int i = 0; i < size(dst); i++) {
    dst(i) = Allreduce<4>::run(src(i), op);
  }
}

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& summary, Operator& op) {
  thread_reduce_<zero_init>(tensor, summary, op);
  quad_allreduce_(summary, summary, op);
}

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& max) {
  MaxOp<float> max_op;
  reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template <bool Scale_max = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0>& tensor, Tensor<Engine1, Layout1> const& max, const float scale) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    // If max is -inf, then all elements must have been -inf (possibly due to masking).
    // We don't want (-inf - (-inf)) since that would give NaN.
    // If we don't have float around M_LOG2E the multiplication is done in fp64.
    const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
      // max * log_2(e)) This allows the compiler to use the ffma
      // instruction instead of fadd and fmul separately.
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
    }
  }
}

// Apply the exp to all the elements.
template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0>& tensor, Tensor<Engine1, Layout1>& max, Tensor<Engine1, Layout1>& sum, const float scale) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    MaxOp<float> max_op;
    max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
#pragma unroll
    for (int ni = 1; ni < size<1>(tensor); ni++) {
      max(mi) = max_op(max(mi), tensor(mi, ni));
    }
    max(mi) = Allreduce<4>::run(max(mi), max_op);
    // If max is -inf, then all elements must have been -inf (possibly due to masking).
    // We don't want (-inf - (-inf)) since that would give NaN.
    const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
    sum(mi) = 0;
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
      // max * log_2(e)) This allows the compiler to use the ffma
      // instruction instead of fadd and fmul separately.
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
      sum(mi) += tensor(mi, ni);
    }
    SumOp<float> sum_op;
    sum(mi) = Allreduce<4>::run(sum(mi), sum_op);
  }
}

template <typename Engine, typename Layout>
inline __device__ void apply_mask(Tensor<Engine, Layout>& tensor, const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
  // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
  static_assert(Layout::rank == 2, "Only support 2D Tensor");
  const int lane_id = threadIdx.x % 32;
  const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
#pragma unroll
  for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
    const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
    for (int j = 0; j < size<1, 0>(tensor); ++j) {
      const int col_idx = col_idx_base + j;
      if (col_idx >= max_seqlen_k) {
// Without the "make_coord" we get wrong results
#pragma unroll
        for (int mi = 0; mi < size<0>(tensor); ++mi) {
          tensor(mi, make_coord(j, nj)) = -INFINITY;
        }
      }
    }
  }
}

template <bool HasWSLeft = true, typename Engine, typename Layout>
inline __device__ void apply_mask_local(Tensor<Engine, Layout>& tensor, const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q, const int warp_row_stride,
                                        const int window_size_left, const int window_size_right) {
  // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
  static_assert(Layout::rank == 2, "Only support 2D Tensor");
  const int lane_id = threadIdx.x % 32;
  // const int row_idx_offset = row_idx_offset_ + lane_id / 4;
  const int row_idx_offset = row_idx_offset_;
  const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
#pragma unroll
  for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
#pragma unroll
    for (int i = 0; i < size<0, 0>(tensor); ++i) {
      const int row_idx = row_idx_base + i * 8;
      const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
      const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
#pragma unroll
      for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
          const int col_idx = col_idx_base + j;
          if (col_idx >= col_idx_limit_right || (HasWSLeft && col_idx < col_idx_limit_left)) {
            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
          }
        }
      }
      // if (cute::thread0()) {
      //     printf("mi = %d, i = %d, row_idx = %d, max_seqlen_k = %d\n", mi, i, row_idx, max_seqlen_k);
      //     print(tensor(make_coord(i, mi), _));
      //     // print(tensor(_, j + nj * size<1, 0>(tensor)));
      // }
    }
  }
}

template <typename Engine, typename Layout>
inline __device__ void apply_mask_causal(Tensor<Engine, Layout>& tensor, const int col_idx_offset_,
                                         const int max_seqlen_k, const int row_idx_offset_,
                                         const int max_seqlen_q, const int warp_row_stride) {
  // Causal masking is equivalent to local masking with window_size_left = infinity and window_size_right = 0
  apply_mask_local</*HasWSLeft=*/false>(tensor, col_idx_offset_, max_seqlen_k, row_idx_offset_,
                                        max_seqlen_q, warp_row_stride, -1, 0);
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void apply_mask_causal_w_idx(
    Tensor<Engine0, Layout0>& tensor, Tensor<Engine1, Layout1> const& idx_rowcol,
    const int col_idx_offset_, const int max_seqlen_k, const int row_idx_offset_) {
  // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 2, "Only support 2D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(idx_rowcol));
  CUTE_STATIC_ASSERT_V(size<1>(tensor) == size<1>(idx_rowcol));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const int col_idx_limit = std::min(max_seqlen_k, 1 + row_idx_offset_ + get<0>(idx_rowcol(mi, 0)));
#pragma unroll
    for (int ni = 0; ni < size<1, 1>(tensor); ++ni) {
      if (col_idx_offset_ + get<1>(idx_rowcol(0, ni)) >= col_idx_limit) {
        tensor(mi, ni) = -INFINITY;
      }
    }
    // if (cute::thread0()) {
    //     printf("ni = %d, j = %d, col_idx = %d, max_seqlen_k = %d\n", ni, j, col_idx, max_seqlen_k);
    //     print(tensor(_, make_coord(j, ni)));
    //     // print(tensor(_, j + ni * size<1, 0>(tensor)));
    // }
  }
}

template <int kNRows>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        if (Is_first) {
            flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            flash::reduce_sum</*zero_init=*/true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            flash::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
    };

    template<bool Is_dropout=false, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
        }
        return lse;
    };
};

template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q,
                                    const int window_size_left, const int window_size_right,
                                    const float alibi_slope=0.f)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , alibi_slope(!Has_alibi ? 0.0 : alibi_slope) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride) {
        static_assert(!(Causal_mask && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || Is_local || !Is_even_MN;
        // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_local=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_local, Is_even_MN, Need_masking); }
        if constexpr (Need_masking) {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;
            const int lane_id = threadIdx.x % 32;
            const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
            if constexpr (Col_idx_only) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            // No causal, no local
                            if constexpr (Has_alibi) {
                                tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                            }
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = row_idx_base + i * 8;
                        const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 8;
                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                const int col_idx = col_idx_base + j;
                                if constexpr (Has_alibi) {
                                    if constexpr (Is_causal) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) += alibi_slope * col_idx;
                                    } else {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);

                                    }
                                }
                                if constexpr (Causal_mask) {
                                    if (col_idx >= col_idx_limit_right) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                                if constexpr (Is_local) {
                                    if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                                if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                    // Causal and Local already handles MN masking
                                    if (col_idx >= max_seqlen_k) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

};

}  // namespace flash
