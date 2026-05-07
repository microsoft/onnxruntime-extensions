// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <vector>
#include <algorithm>

#include "ext_status.h"
#include "image_resample_float.h"
#include "op_def_struct.h"

namespace ort_extensions {

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
      int64_t source_height, int64_t source_width,
      int64_t patch_size, int64_t max_patches, int64_t pooling_kernel_size) {
    const int64_t side_mult = pooling_kernel_size * patch_size;
    const double target_px = static_cast<double>(max_patches) * patch_size * patch_size;
    const double total_px = static_cast<double>(source_height) * source_width;
    const double factor = std::sqrt(target_px / total_px);

    int64_t target_height = static_cast<int64_t>(std::floor(factor * source_height / side_mult)) * side_mult;
    int64_t target_width = static_cast<int64_t>(std::floor(factor * source_width / side_mult)) * side_mult;

    const int64_t max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;

    // Match HuggingFace behavior: reject images where both dimensions round
    // to zero (extreme aspect ratio).  See image_processing_gemma4.py L61.
    if (target_height == 0 && target_width == 0) {
      return {0, 0};  // caller checks and returns an error
    } else if (target_height == 0) {
      target_height = side_mult;
      target_width = std::min(
          static_cast<int64_t>(std::floor(static_cast<double>(source_width) / source_height)) * side_mult,
          max_side);
      if (target_width == 0) target_width = side_mult;
    } else if (target_width == 0) {
      target_width = side_mult;
      target_height = std::min(
          static_cast<int64_t>(std::floor(static_cast<double>(source_height) / source_width)) * side_mult,
          max_side);
      if (target_height == 0) target_height = side_mult;
    }

    return {target_height, target_width};
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

    const int64_t source_height = dims[0];
    const int64_t source_width = dims[1];
    constexpr int64_t C = 3;

    // --- aspect-ratio-preserving resize ----------------------------------
    const int64_t max_patches = max_soft_tokens_ * pooling_kernel_size_ * pooling_kernel_size_;
    auto [target_height, target_width] = GetAspectRatioPreservingSize(
        source_height, source_width, patch_size_, max_patches, pooling_kernel_size_);

    if (target_height == 0 || target_width == 0) {
      return {kOrtxErrorInvalidArgument,
          "[Gemma4ImageTransform]: image has extreme aspect ratio ("
          + std::to_string(source_height) + ", " + std::to_string(source_width)
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
    const uint8_t* source_data = input.Data();

    // Bicubic resize uint8 → float with fused rescale
    std::vector<float> destination_float(static_cast<size_t>(target_height) * target_width * C);
    BicubicResizeU8ToFloatRGB(
        destination_float.data(), source_data,
        static_cast<int>(source_height), static_cast<int>(source_width),
        static_cast<int>(target_height), static_cast<int>(target_width));

    // --- patchify ---------------------------------------------------------
    const int64_t ph = target_height / patch_size_;   // patches along height
    const int64_t pw = target_width / patch_size_;    // patches along width
    const int64_t num_patches = ph * pw;
    const int64_t patch_dim = patch_size_ * patch_size_ * C;   // 16*16*3 = 768

    // Output 0: pixel_values  (max_patches, patch_dim), zero-padded
    float* pv = pixel_values_out.Allocate({max_patches, patch_dim});
    std::memset(pv, 0, static_cast<size_t>(max_patches * patch_dim) * sizeof(float));

    for (int64_t py = 0; py < ph; ++py) {
      for (int64_t px = 0; px < pw; ++px) {
        const int64_t patch_idx = py * pw + px;
        float* destination = pv + patch_idx * patch_dim;
        // Copy patch pixels in HWC raster order to match HuggingFace:
        // permute(1, 3, 2, 4, 0) in convert_image_to_patches() yields
        // (patch_height, patch_width, num_channels) within each patch.
        for (int64_t dy = 0; dy < patch_size_; ++dy) {
          const int64_t row = py * patch_size_ + dy;
          for (int64_t dx = 0; dx < patch_size_; ++dx) {
            const int64_t col = px * patch_size_ + dx;
            const int64_t base = (dy * patch_size_ + dx) * C;
            const float* pixel = destination_float.data() + (row * target_width + col) * C;
            destination[base + 0] = pixel[0];
            destination[base + 1] = pixel[1];
            destination[base + 2] = pixel[2];
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
