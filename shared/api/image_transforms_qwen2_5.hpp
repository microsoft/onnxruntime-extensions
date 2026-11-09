// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include "ext_status.h"
#include "op_def_struct.h"

namespace ort_extensions {

class PatchImage {
 public:
  PatchImage() = default;

  OrtxStatus Compute(const ortc::Tensor<float>& input, ortc::Tensor<float>& output,
                     ortc::Tensor<int64_t>& grid_thw_output) {
    const auto& dims = input.Shape();
    if (dims.size() != 3 && dims.size() != 4) {
      return {kOrtxErrorInvalidArgument, "[PatchImage]: input must be HWC or THWC float"};
    }
    const size_t offset = dims.size() - 3;
    const int64_t frames = offset == 0 ? 1 : dims[0];
    const int64_t height = dims[offset];
    const int64_t width = dims[offset + 1];
    const int64_t channels = dims[offset + 2];
    if (frames <= 0 || height <= 0 || width <= 0 || channels <= 0 || patch_size_ <= 0 || temporal_patch_size_ <= 0 ||
        merge_size_ <= 0 || height % patch_size_ != 0 || width % patch_size_ != 0 ||
        (height / patch_size_) % merge_size_ != 0 || (width / patch_size_) % merge_size_ != 0) {
      return {kOrtxErrorInvalidArgument, "[PatchImage]: invalid frame, patch or merge geometry"};
    }
    const int64_t grid_t = frames / temporal_patch_size_ + (frames % temporal_patch_size_ != 0);
    const int64_t grid_h = height / patch_size_;
    const int64_t grid_w = width / patch_size_;
    auto product = [](std::initializer_list<int64_t> factors, int64_t& value) {
      value = 1;
      for (auto factor : factors) {
        if (factor > (std::numeric_limits<int64_t>::max)() / value) {
          return false;
        }
        value *= factor;
      }
      return static_cast<uint64_t>(value) <= (std::numeric_limits<size_t>::max)() / sizeof(float);
    };
    int64_t input_elements, out_rows, out_cols, output_elements;
    if (!product({frames, height, width, channels}, input_elements) || !product({grid_t, grid_h, grid_w}, out_rows) ||
        !product({channels, temporal_patch_size_, patch_size_, patch_size_}, out_cols) ||
        !product({out_rows, out_cols}, output_elements)) {
      return {kOrtxErrorInvalidArgument, "[PatchImage]: patch storage overflows"};
    }
    float* destination = output.Allocate({out_rows, out_cols});
    const float* source = input.Data();
    // Merge-block patch order, with channel/temporal/spatial elements inside each row.
    // Only the incomplete final temporal group repeats the last real frame.
    for (int64_t t = 0; t < grid_t; ++t)
      for (int64_t block_y = 0; block_y < grid_h / merge_size_; ++block_y)
        for (int64_t block_x = 0; block_x < grid_w / merge_size_; ++block_x)
          for (int64_t merge_y = 0; merge_y < merge_size_; ++merge_y)
            for (int64_t merge_x = 0; merge_x < merge_size_; ++merge_x)
              for (int64_t channel = 0; channel < channels; ++channel)
                for (int64_t temporal = 0; temporal < temporal_patch_size_; ++temporal)
                  for (int64_t pixel_y = 0; pixel_y < patch_size_; ++pixel_y)
                    for (int64_t pixel_x = 0; pixel_x < patch_size_; ++pixel_x) {
                      const int64_t frame = (std::min)(t * temporal_patch_size_ + temporal, frames - 1);
                      const int64_t y = (block_y * merge_size_ + merge_y) * patch_size_ + pixel_y;
                      const int64_t x = (block_x * merge_size_ + merge_x) * patch_size_ + pixel_x;
                      *destination++ = source[((frame * height + y) * width + x) * channels + channel];
                    }
    int64_t* grid = grid_thw_output.Allocate({1, 3});
    grid[0] = grid_t;
    grid[1] = grid_h;
    grid[2] = grid_w;
    return {};
  }

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "patch_size") {
        patch_size_ = std::get<int64_t>(value);
      } else if (key == "temporal_patch_size") {
        temporal_patch_size_ = std::get<int64_t>(value);
      } else if (key == "merge_size") {
        merge_size_ = std::get<int64_t>(value);
      } else {
        return {kOrtxErrorInvalidArgument, "[PatchImage]: Invalid argument " + key};
      }
    }
    return {};
  }

 private:
  // Constants set to defaults
  int64_t patch_size_{14};
  int64_t temporal_patch_size_{2};
  int64_t merge_size_{2};
};

}  // namespace ort_extensions
