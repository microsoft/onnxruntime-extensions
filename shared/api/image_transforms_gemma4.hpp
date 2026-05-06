// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

#include "ext_status.h"
#include "op_def_struct.h"

namespace ort_extensions {

// ---------------------------------------------------------------------------
// Float-domain 3-channel bicubic resize with antialias.
//
// Matches torchvision F.resize(BICUBIC, antialias=True) / Pillow "F" mode:
//   same Keys cubic kernel (a=-0.5), same coefficient precomputation,
//   same separable horizontal→vertical pass order, NO clamping (overshoot
//   allowed).  Processes all 3 channels in each inner-loop iteration for
//   cache-friendly packed-float operation.
// ---------------------------------------------------------------------------
namespace detail {

inline double bicubic_kernel(double x) {
  // Keys cubic, a = -0.5
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((1.5 * x - 2.5) * x) * x + 1.0;
  if (x < 2.0) return (((-0.5 * x + 2.5) * x - 4.0) * x + 2.0);
  return 0.0;
}

// Precompute filter coefficients for one dimension.
// Returns kernel size; fills bounds (xmin, count per output pixel)
// and kk (normalised weights).  Mirrors Pillow's precompute_coeffs().
inline int precompute_coeffs(
    int in_size, double in0, double in1, int out_size,
    std::vector<int>& bounds, std::vector<double>& kk) {
  constexpr double kSupport = 2.0;  // bicubic support

  double scale = (in1 - in0) / out_size;
  double filter_scale = std::max(scale, 1.0);
  double support = kSupport * filter_scale;
  int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

  bounds.resize(static_cast<size_t>(out_size) * 2);
  kk.resize(static_cast<size_t>(out_size) * ksize);

  double inv_filter_scale = 1.0 / filter_scale;
  for (int xx = 0; xx < out_size; ++xx) {
    double center = in0 + (xx + 0.5) * scale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;

    double* k = &kk[static_cast<size_t>(xx) * ksize];
    double ww = 0.0;
    for (int x = 0; x < xmax; ++x) {
      double w = bicubic_kernel((x + xmin - center + 0.5) * inv_filter_scale);
      k[x] = w;
      ww += w;
    }
    if (ww != 0.0) {
      for (int x = 0; x < xmax; ++x) k[x] /= ww;
    }
    for (int x = xmax; x < ksize; ++x) k[x] = 0.0;
    bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
    bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
  }
  return ksize;
}

// Horizontal resample from uint8 source: (in_h, in_w, 3) uint8 → (in_h, out_w, 3) float.
// Fuses the uint8→float rescale (1/255) into the filter accumulation to avoid
// allocating a full-resolution float copy of the source image.
inline void resample_horizontal_rgb_u8(
    float* dst, const uint8_t* src,
    int in_h, int in_w, int out_w,
    int ksize, const int* bounds, const double* kk) {
  constexpr double kRescale = 1.0 / 255.0;
  for (int y = 0; y < in_h; ++y) {
    const uint8_t* src_row = src + static_cast<size_t>(y) * in_w * 3;
    float* dst_row = dst + static_cast<size_t>(y) * out_w * 3;
    for (int xx = 0; xx < out_w; ++xx) {
      int xmin = bounds[xx * 2];
      int xmax = bounds[xx * 2 + 1];
      const double* k = &kk[static_cast<size_t>(xx) * ksize];
      double s0 = 0.0, s1 = 0.0, s2 = 0.0;
      for (int x = 0; x < xmax; ++x) {
        const uint8_t* p = src_row + (xmin + x) * 3;
        s0 += p[0] * k[x];
        s1 += p[1] * k[x];
        s2 += p[2] * k[x];
      }
      dst_row[xx * 3 + 0] = static_cast<float>(s0 * kRescale);
      dst_row[xx * 3 + 1] = static_cast<float>(s1 * kRescale);
      dst_row[xx * 3 + 2] = static_cast<float>(s2 * kRescale);
    }
  }
}

// Horizontal resample: (in_h, in_w, 3) → (in_h, out_w, 3)
inline void resample_horizontal_rgb(
    float* dst, const float* src,
    int in_h, int in_w, int out_w,
    int ksize, const int* bounds, const double* kk) {
  for (int y = 0; y < in_h; ++y) {
    const float* src_row = src + static_cast<size_t>(y) * in_w * 3;
    float* dst_row = dst + static_cast<size_t>(y) * out_w * 3;
    for (int xx = 0; xx < out_w; ++xx) {
      int xmin = bounds[xx * 2];
      int xmax = bounds[xx * 2 + 1];
      const double* k = &kk[static_cast<size_t>(xx) * ksize];
      double s0 = 0.0, s1 = 0.0, s2 = 0.0;
      for (int x = 0; x < xmax; ++x) {
        const float* p = src_row + (xmin + x) * 3;
        s0 += p[0] * k[x];
        s1 += p[1] * k[x];
        s2 += p[2] * k[x];
      }
      dst_row[xx * 3 + 0] = static_cast<float>(s0);
      dst_row[xx * 3 + 1] = static_cast<float>(s1);
      dst_row[xx * 3 + 2] = static_cast<float>(s2);
    }
  }
}

// Vertical resample: (in_h, w, 3) → (out_h, w, 3)
inline void resample_vertical_rgb(
    float* dst, const float* src,
    int in_h, int w, int out_h,
    int ksize, const int* bounds, const double* kk) {
  for (int yy = 0; yy < out_h; ++yy) {
    int ymin = bounds[yy * 2];
    int ymax = bounds[yy * 2 + 1];
    const double* k = &kk[static_cast<size_t>(yy) * ksize];
    float* dst_row = dst + static_cast<size_t>(yy) * w * 3;
    for (int xx = 0; xx < w; ++xx) {
      double s0 = 0.0, s1 = 0.0, s2 = 0.0;
      for (int y = 0; y < ymax; ++y) {
        const float* p = src + (static_cast<size_t>(ymin + y) * w + xx) * 3;
        s0 += p[0] * k[y];
        s1 += p[1] * k[y];
        s2 += p[2] * k[y];
      }
      dst_row[xx * 3 + 0] = static_cast<float>(s0);
      dst_row[xx * 3 + 1] = static_cast<float>(s1);
      dst_row[xx * 3 + 2] = static_cast<float>(s2);
    }
  }
}

}  // namespace detail

// Bicubic resize from uint8 RGB to float RGB: (src_h, src_w, 3) uint8 → (tgt_h, tgt_w, 3) float.
// Fuses the 1/255 rescale into the horizontal pass to avoid a full-resolution
// float copy of the source image. Output may exceed [0,1] due to bicubic
// overshoot (matching torchvision F.resize behavior).
inline void BicubicResizeU8ToFloatRGB(
    float* dst,
    const uint8_t* src, int src_h, int src_w,
    int tgt_h, int tgt_w) {
  // Precompute coefficients for each dimension
  std::vector<int> bounds_h, bounds_v;
  std::vector<double> kk_h, kk_v;
  int ksize_h = detail::precompute_coeffs(
      src_w, 0.0, static_cast<double>(src_w), tgt_w, bounds_h, kk_h);
  int ksize_v = detail::precompute_coeffs(
      src_h, 0.0, static_cast<double>(src_h), tgt_h, bounds_v, kk_v);

  bool need_h = (tgt_w != src_w);
  bool need_v = (tgt_h != src_h);

  if (need_h && need_v) {
    // Horizontal pass reads uint8, writes float (fused rescale).
    // Only resample rows that the vertical pass will read.
    int ybox_first = bounds_v[0];
    int ybox_last = bounds_v[static_cast<size_t>(tgt_h - 1) * 2]
                  + bounds_v[static_cast<size_t>(tgt_h - 1) * 2 + 1];
    int temp_h = ybox_last - ybox_first;

    std::vector<float> temp(static_cast<size_t>(temp_h) * tgt_w * 3);
    detail::resample_horizontal_rgb_u8(
        temp.data(), src + static_cast<size_t>(ybox_first) * src_w * 3,
        temp_h, src_w, tgt_w, ksize_h, bounds_h.data(), kk_h.data());

    // Shift vertical bounds to account for ybox_first offset
    for (int i = 0; i < tgt_h; ++i) {
      bounds_v[static_cast<size_t>(i) * 2] -= ybox_first;
    }

    // Vertical pass: (temp_h, tgt_w, 3) float → (tgt_h, tgt_w, 3) float
    detail::resample_vertical_rgb(
        dst, temp.data(), temp_h, tgt_w, tgt_h,
        ksize_v, bounds_v.data(), kk_v.data());
  } else if (need_h) {
    detail::resample_horizontal_rgb_u8(
        dst, src, src_h, src_w, tgt_w, ksize_h, bounds_h.data(), kk_h.data());
  } else if (need_v) {
    // Rare: same width, different height — convert to float first
    std::vector<float> src_float(static_cast<size_t>(src_h) * src_w * 3);
    constexpr float kRescale = 1.0f / 255.0f;
    for (size_t i = 0; i < src_float.size(); ++i)
      src_float[i] = static_cast<float>(src[i]) * kRescale;
    detail::resample_vertical_rgb(
        dst, src_float.data(), src_h, src_w, tgt_h,
        ksize_v, bounds_v.data(), kk_v.data());
  } else {
    constexpr float kRescale = 1.0f / 255.0f;
    for (size_t i = 0; i < static_cast<size_t>(src_h) * src_w * 3; ++i)
      dst[i] = static_cast<float>(src[i]) * kRescale;
  }
}

// Gemma 4 vision preprocessing: aspect-ratio-preserving resize, patchification,
// and 2-D position-ID computation that matches the HuggingFace
// Gemma4ImageProcessor exactly.
//
// Pipeline:  DecodeImage  ->  Gemma4ImageTransform
//
// Inputs:   uint8  (H, W, 3)   — decoded RGB image
// Outputs:  float  (max_patches, patch_size*patch_size*3) — patchified pixel values (may exceed [0, 1] due to bicubic overshoot)
//           int64  (max_patches, 2)                        — 2-D position IDs  (x=col, y=row)
//           int64  (1,)                                    — number of soft tokens for this image
class Gemma4ImageTransform {
 public:
  Gemma4ImageTransform() = default;

  // Compute the target (height, width) that preserves the source aspect ratio
  // while fitting within the patch budget.  Both dimensions are rounded DOWN to
  // the nearest multiple of `side_mult = pooling_kernel_size * patch_size`.
  static std::pair<int64_t, int64_t> GetAspectRatioPreservingSize(
      int64_t src_h, int64_t src_w,
      int64_t patch_size, int64_t max_patches, int64_t pooling_kernel_size) {
    const int64_t side_mult = pooling_kernel_size * patch_size;
    const double target_px = static_cast<double>(max_patches) * patch_size * patch_size;
    const double total_px = static_cast<double>(src_h) * src_w;
    const double factor = std::sqrt(target_px / total_px);

    int64_t tgt_h = static_cast<int64_t>(std::floor(factor * src_h / side_mult)) * side_mult;
    int64_t tgt_w = static_cast<int64_t>(std::floor(factor * src_w / side_mult)) * side_mult;

    const int64_t max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;

    // Match HuggingFace behavior: reject images where both dimensions round
    // to zero (extreme aspect ratio).  See image_processing_gemma4.py L61.
    if (tgt_h == 0 && tgt_w == 0) {
      return {0, 0};  // caller checks and returns an error
    } else if (tgt_h == 0) {
      tgt_h = side_mult;
      tgt_w = std::min(
          static_cast<int64_t>(std::floor(static_cast<double>(src_w) / src_h)) * side_mult,
          max_side);
      if (tgt_w == 0) tgt_w = side_mult;
    } else if (tgt_w == 0) {
      tgt_w = side_mult;
      tgt_h = std::min(
          static_cast<int64_t>(std::floor(static_cast<double>(src_h) / src_w)) * side_mult,
          max_side);
      if (tgt_h == 0) tgt_h = side_mult;
    }

    return {tgt_h, tgt_w};
  }

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input,
                     ortc::Tensor<float>& pixel_values_out,
                     ortc::Tensor<int64_t>& position_ids_out,
                     ortc::Tensor<int64_t>& num_soft_tokens_out) {
    // --- validate input --------------------------------------------------
    const auto& dims = input.Shape();
    if (dims.size() != 3ULL || dims[2] != 3) {
      return {kOrtxErrorInvalidArgument,
              "[Gemma4ImageTransform]: expected (H, W, 3) uint8 input"};
    }

    const int64_t src_h = dims[0];
    const int64_t src_w = dims[1];
    constexpr int64_t C = 3;

    // --- aspect-ratio-preserving resize ----------------------------------
    const int64_t max_patches = max_soft_tokens_ * pooling_kernel_size_ * pooling_kernel_size_;
    auto [tgt_h, tgt_w] = GetAspectRatioPreservingSize(
        src_h, src_w, patch_size_, max_patches, pooling_kernel_size_);

    if (tgt_h == 0 || tgt_w == 0) {
      return {kOrtxErrorInvalidArgument,
          "[Gemma4ImageTransform]: image has extreme aspect ratio ("
          + std::to_string(src_h) + ", " + std::to_string(src_w)
          + ") and cannot be resized to fit the patch constraints"};
    }

    // Use float-domain bicubic resampling to match torchvision's
    // F.resize(antialias=True) used by HuggingFace.  The uint8 ("RGB") path
    // clips bicubic overshoot to [0,255] via clip8(), but torchvision operates
    // on float32 tensors where bicubic can produce values outside [0,1].
    // Our packed-float RGB resize uses the same Pillow/torchvision kernel
    // (Keys cubic, a=-0.5) with antialias and no clamping.
    // The horizontal pass fuses the uint8→float 1/255 rescale to avoid
    // allocating a full-resolution float copy of the source image.
    const uint8_t* src_data = input.Data();

    // Bicubic resize uint8 → float with fused rescale
    std::vector<float> dst_float(static_cast<size_t>(tgt_h) * tgt_w * C);
    BicubicResizeU8ToFloatRGB(
        dst_float.data(), src_data,
        static_cast<int>(src_h), static_cast<int>(src_w),
        static_cast<int>(tgt_h), static_cast<int>(tgt_w));

    // --- patchify ---------------------------------------------------------
    const int64_t ph = tgt_h / patch_size_;   // patches along height
    const int64_t pw = tgt_w / patch_size_;   // patches along width
    const int64_t num_patches = ph * pw;
    const int64_t patch_dim = patch_size_ * patch_size_ * C;   // 16*16*3 = 768

    // Output 0: pixel_values  (max_patches, patch_dim), zero-padded
    float* pv = pixel_values_out.Allocate({max_patches, patch_dim});
    std::memset(pv, 0, static_cast<size_t>(max_patches * patch_dim) * sizeof(float));

    for (int64_t py = 0; py < ph; ++py) {
      for (int64_t px = 0; px < pw; ++px) {
        const int64_t patch_idx = py * pw + px;
        float* dst = pv + patch_idx * patch_dim;
        // Copy patch pixels in HWC raster order to match HuggingFace:
        // permute(1, 3, 2, 4, 0) in convert_image_to_patches() yields
        // (patch_height, patch_width, num_channels) within each patch.
        for (int64_t dy = 0; dy < patch_size_; ++dy) {
          const int64_t row = py * patch_size_ + dy;
          for (int64_t dx = 0; dx < patch_size_; ++dx) {
            const int64_t col = px * patch_size_ + dx;
            const int64_t base = (dy * patch_size_ + dx) * C;
            const float* pixel = dst_float.data() + (row * tgt_w + col) * C;
            dst[base + 0] = pixel[0];
            dst[base + 1] = pixel[1];
            dst[base + 2] = pixel[2];
          }
        }
      }
    }

    // Output 1: position_ids  (max_patches, 2)
    // Real patches get (x=col, y=row); padding gets (-1, -1).
    int64_t* pos = position_ids_out.Allocate({max_patches, 2});
    for (int64_t py = 0; py < ph; ++py) {
      for (int64_t px = 0; px < pw; ++px) {
        const int64_t idx = (py * pw + px) * 2;
        pos[idx]     = px;   // x = column
        pos[idx + 1] = py;   // y = row
      }
    }
    // Padding positions
    for (int64_t i = num_patches; i < max_patches; ++i) {
      pos[i * 2]     = -1;
      pos[i * 2 + 1] = -1;
    }

    // Output 2: num_soft_tokens  (1,) — the number of vision tokens after pooling
    int64_t* nst = num_soft_tokens_out.Allocate({1});
    nst[0] = num_patches / (pooling_kernel_size_ * pooling_kernel_size_);

    return {};
  }

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "patch_size") {
        patch_size_ = std::get<int64_t>(value);
      } else if (key == "max_soft_tokens") {
        max_soft_tokens_ = std::get<int64_t>(value);
      } else if (key == "pooling_kernel_size") {
        pooling_kernel_size_ = std::get<int64_t>(value);
      } else {
        return {kOrtxErrorInvalidArgument,
                "[Gemma4ImageTransform]: unknown attribute '" + key + "'"};
      }
    }
    return {};
  }

 private:
  int64_t patch_size_ = 16;
  int64_t max_soft_tokens_ = 280;
  int64_t pooling_kernel_size_ = 3;
};

}  // namespace ort_extensions
