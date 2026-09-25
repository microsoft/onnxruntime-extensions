// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstring>
#include "ext_status.h"
#include "op_def_struct.h"

namespace ort_extensions {

/// Extracts image dimensions from a CHW float tensor and passes the pixel data through unchanged.
///
/// Input:  [C, H, W]  float tensor (output of Permute3D)
/// Output[0]: [C, H, W]  float tensor — pixel values, unchanged
/// Output[1]: [2]         int64 tensor — [H, W] image dimensions
///
/// When processing N images the framework's StackTensors produces:
///   pixel_values  [N, C, max_H, max_W]  (zero-padded for different sizes)
///   image_sizes   [N, 2]
class PixtralImageSizes {
 public:
  PixtralImageSizes() = default;

  OrtxStatus Compute(const ortc::Tensor<float>& input,
                     ortc::Tensor<float>& output,
                     ortc::Tensor<int64_t>& image_sizes_output) {
    const auto& dims = input.Shape();
    if (dims.size() != 3ULL) {
      return {kOrtxErrorInvalidArgument, "[PixtralImageSizes]: input must be a 3D CHW float tensor"};
    }

    int64_t C = dims[0];
    int64_t H = dims[1];
    int64_t W = dims[2];

    if (C != 3) {
      return {kOrtxErrorInvalidArgument, "[PixtralImageSizes]: expected C=3 (RGB), got C=" + std::to_string(C)};
    }

    // Pass through pixel values unchanged
    const float* src = input.Data();
    const int64_t num_elems = C * H * W;
    output.Allocate({C, H, W});
    float* dst = const_cast<float*>(output.Data());
    std::memcpy(dst, src, static_cast<size_t>(num_elems) * sizeof(float));

    // Emit image dimensions as a flat [2] tensor (StackTensors prepends the batch dim)
    image_sizes_output.Allocate({2});
    int64_t* sizes = const_cast<int64_t*>(image_sizes_output.Data());
    sizes[0] = H;
    sizes[1] = W;

    return {};
  }

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& attr : attrs) {
      return {kOrtxErrorInvalidArgument, "[PixtralImageSizes]: unexpected attribute " + attr.first};
    }
    return {};
  }
};

}  // namespace ort_extensions
