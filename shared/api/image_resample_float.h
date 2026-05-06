// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Float-domain bicubic resize with antialias for 3-channel (RGB) images.
//
// Matches torchvision F.resize(BICUBIC, antialias=True) / Pillow "F" mode:
//   same Keys cubic kernel (a=-0.5), same coefficient precomputation,
//   same separable horizontal→vertical pass order, NO uint8 clamping
//   (overshoot allowed).
//
// This is a shared utility — any image transform that needs float-domain
// bicubic resize can include this header instead of duplicating the code.
// Currently used by Gemma4; available for phi3, phi4, mllama, etc. if they
// need to match torchvision behavior precisely.

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace ort_extensions {
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
//
// Parameters:
//   dst  — caller-allocated buffer of size tgt_h * tgt_w * 3 floats.
//          Output is HWC-interleaved float RGB, nominally in [0,1] but
//          values may slightly exceed this range due to bicubic overshoot.
//   src  — source image in HWC uint8 RGB layout (src_h * src_w * 3 bytes).
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

}  // namespace ort_extensions
