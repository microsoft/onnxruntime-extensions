// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <vector>
#include "ext_status.h"
#include "op_def_struct.h"

namespace ort_extensions {

class PatchImage {
 public:
  PatchImage() = default;

  OrtxStatus Compute(const ortc::Tensor<float>& input, ortc::Tensor<float>& output) {
    // Validate and read HWC
    const auto& dims = input.Shape();
    if (dims.size() != 3ULL) {
      return {kOrtxErrorInvalidArgument, "[PatchImage]: input must be HWC float"};
    }

    int64_t H = dims[0];
    int64_t W = dims[1];
    int64_t C = dims[2];

    const float* data = input.Data();
    const int64_t total_elems = H * W * C;

    // Convert RGB HWC -> CHW (channel 0 = R, 1 = G, 2 = B)
    std::vector<float> chw;
    chw.reserve(static_cast<size_t>(H) * static_cast<size_t>(W) * static_cast<size_t>(C));
    for (int64_t ch = 0; ch < C; ++ch) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          size_t base = static_cast<size_t>((h * W + w) * C);
          // Input is RGB so base+0=R, base+1=G, base+2=B
          chw.push_back(data[base + ch]);
        }
      }
    }

    // Add batch dimension (frames = 1 initially) and prepare patches vector
    std::vector<float> patches = chw;
    int64_t frames = 1;

    // Perform temporal padding to make frames % temporal_patch_size_ == 0
    if (frames % temporal_patch_size_ != 0) {
      int64_t pad = temporal_patch_size_ - (frames % temporal_patch_size_);
      for (int64_t i = 0; i < pad; ++i) {
        patches.insert(patches.end(), chw.begin(), chw.end());
      }
      frames += pad;
    }

    // Compute grid sizes
    int64_t grid_t = frames / temporal_patch_size_;
    int64_t grid_h = H / patch_size_;
    int64_t grid_w = W / patch_size_;

    // Reshape dimensions (Python 9D)
    if (merge_size_ <= 0) {
      return {kOrtxErrorInvalidArgument, "[PatchImage]: merge_size must be > 0"};
    }
    int64_t GH2 = grid_h / merge_size_;
    int64_t GW2 = grid_w / merge_size_;

    const int64_t expected =
        grid_t * temporal_patch_size_ * C *
        GH2 * merge_size_ * patch_size_ *
        GW2 * merge_size_ * patch_size_;

    if (expected != static_cast<int64_t>(patches.size())) {
      return {kOrtxErrorInvalidArgument, "Patch reshape mismatch"};
    }

    // Allocate memory for reshaped data and fill in Python reshape order
    std::vector<float> reshaped;
    reshaped.resize(static_cast<size_t>(expected));

    auto idx9 = [&](int64_t t, int64_t tp, int64_t c, int64_t gh2, int64_t m1,
                    int64_t ph, int64_t gw2, int64_t m2, int64_t pw) -> int64_t {
      int64_t v = t;
      v = v * temporal_patch_size_ + tp;
      v = v * C + c;
      v = v * GH2 + gh2;
      v = v * merge_size_ + m1;
      v = v * patch_size_ + ph;
      v = v * GW2 + gw2;
      v = v * merge_size_ + m2;
      v = v * patch_size_ + pw;
      return v;
    };

    {
      size_t p = 0;
      for (int64_t t = 0; t < grid_t; ++t)
        for (int64_t tp = 0; tp < temporal_patch_size_; ++tp)
          for (int64_t c = 0; c < C; ++c)
            for (int64_t gh2 = 0; gh2 < GH2; ++gh2)
              for (int64_t m1 = 0; m1 < merge_size_; ++m1)
                for (int64_t ph = 0; ph < patch_size_; ++ph)
                  for (int64_t gw2 = 0; gw2 < GW2; ++gw2)
                    for (int64_t m2 = 0; m2 < merge_size_; ++m2)
                      for (int64_t pw = 0; pw < patch_size_; ++pw) {
                        int64_t dst = idx9(t, tp, c, gh2, m1, ph, gw2, m2, pw);
                        reshaped[static_cast<size_t>(dst)] = patches[p++];
                      }
    }

    // Transpose to Python-expected order (0,3,6,4,7,2,1,5,8)
    std::vector<float> transposed;
    transposed.resize(static_cast<size_t>(expected));

    auto idxT = [&](int64_t t, int64_t gh2, int64_t gw2, int64_t m1, int64_t m2,
                    int64_t c, int64_t tp, int64_t ph, int64_t pw) -> int64_t {
      int64_t v = t;
      v = v * GH2 + gh2;
      v = v * GW2 + gw2;
      v = v * merge_size_ + m1;
      v = v * merge_size_ + m2;
      v = v * C + c;
      v = v * temporal_patch_size_ + tp;
      v = v * patch_size_ + ph;
      v = v * patch_size_ + pw;
      return v;
    };

    {
      size_t src_idx = 0;
      for (int64_t t = 0; t < grid_t; ++t)
        for (int64_t tp = 0; tp < temporal_patch_size_; ++tp)
          for (int64_t c = 0; c < C; ++c)
            for (int64_t gh2 = 0; gh2 < GH2; ++gh2)
              for (int64_t m1 = 0; m1 < merge_size_; ++m1)
                for (int64_t ph = 0; ph < patch_size_; ++ph)
                  for (int64_t gw2 = 0; gw2 < GW2; ++gw2)
                    for (int64_t m2 = 0; m2 < merge_size_; ++m2)
                      for (int64_t pw = 0; pw < patch_size_; ++pw) {
                        int64_t dst = idxT(t, gh2, gw2, m1, m2, c, tp, ph, pw);
                        transposed[static_cast<size_t>(dst)] = reshaped[src_idx++];
                      }
    }

    // Final flatten
    int64_t out_rows = grid_t * grid_h * grid_w;
    int64_t out_cols = C * temporal_patch_size_ * patch_size_ * patch_size_;

    output.Allocate({out_rows, out_cols});
    float* outptr = const_cast<float*>(output.Data());
    std::copy(transposed.begin(), transposed.end(), outptr);

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
