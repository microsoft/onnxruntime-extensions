// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

#include "ext_status.h"
#include "op_def_struct.h"
#include "image_resample.h"

namespace ort_extensions {

// Gemma 4 vision preprocessing: aspect-ratio-preserving resize, patchification,
// and 2-D position-ID computation that matches the HuggingFace
// Gemma4ImageProcessor exactly.
//
// Pipeline:  DecodeImage  ->  Gemma4ImageTransform
//
// Inputs:   uint8  (H, W, 3)   — decoded RGB image
// Outputs:  float  (max_patches, patch_size*patch_size*3) — patchified pixel values in [0, 1]
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

    // Use Pillow-style bicubic resampling (same as existing Resize kernel).
    Imaging rgb_src = ImagingNew("RGB", static_cast<int>(src_w), static_cast<int>(src_h));
    if (!rgb_src) {
      return {kOrtxErrorInternal, "[Gemma4ImageTransform]: ImagingNew failed"};
    }

    const uint8_t* src_data = input.Data();
    for (int64_t r = 0; r < src_h; ++r) {
      for (int64_t c = 0; c < src_w; ++c) {
        uint8_t* pixel = reinterpret_cast<uint8_t*>(rgb_src->image[r] + c * 4);
        const auto idx = (r * src_w + c) * C;
        pixel[0] = src_data[idx];
        pixel[1] = src_data[idx + 1];
        pixel[2] = src_data[idx + 2];
        pixel[3] = 0;
      }
    }

    float box[4] = {0.0f, 0.0f, static_cast<float>(src_w), static_cast<float>(src_h)};
    Imaging rgb_dst = ImagingResample(rgb_src, static_cast<int>(tgt_w),
                                      static_cast<int>(tgt_h),
                                      IMAGING_TRANSFORM_BICUBIC, box);
    ImagingDelete(rgb_src);
    if (!rgb_dst) {
      return {kOrtxErrorInternal, "[Gemma4ImageTransform]: ImagingResample failed"};
    }

    // --- patchify ---------------------------------------------------------
    const int64_t ph = tgt_h / patch_size_;   // patches along height
    const int64_t pw = tgt_w / patch_size_;   // patches along width
    const int64_t num_patches = ph * pw;
    const int64_t patch_dim = patch_size_ * patch_size_ * C;   // 16*16*3 = 768
    const float rescale = 1.0f / 255.0f;

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
            uint8_t* pixel = reinterpret_cast<uint8_t*>(
                rgb_dst->image[row] + col * 4);
            const int64_t base = (dy * patch_size_ + dx) * C;
            for (int64_t ch = 0; ch < C; ++ch) {
              dst[base + ch] = static_cast<float>(pixel[ch]) * rescale;
            }
          }
        }
      }
    }
    ImagingDelete(rgb_dst);

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
