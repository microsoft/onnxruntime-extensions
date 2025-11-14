#pragma once

#include <cmath>
#include <vector>
#include "ext_status.h"
#include "op_def_struct.h"

namespace ort_extensions {

class PatchImage {
 public:
  PatchImage() = default;

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<float>& output) {
    const auto& dims = input.Shape();
    int64_t height = dims[0];
    int64_t width = dims[1];
    int64_t channel = dims[2];  // Typically 3 for RGB

    int64_t patch_size = 14;  // From your provided code
    int64_t temporal_patch_size = 2;
    int64_t merge_size = 2;

    // Ensure the image is in the correct format
    const auto image = input.Data();
    std::vector<int64_t> patch_shape = {height / patch_size, width / patch_size};

    // Create patches based on the resized image
    std::vector<float> patches;
    patches.reserve(static_cast<size_t>(height * width)); // Allocate sufficient space

    // Patch extraction logic (like in your Python code)
    for (int64_t i = 0; i < height; i += patch_size) {
      for (int64_t j = 0; j < width; j += patch_size) {
        for (int64_t c = 0; c < channel; ++c) {
          for (int64_t ph = 0; ph < patch_size; ++ph) {
            for (int64_t pw = 0; pw < patch_size; ++pw) {
              int64_t idx = (i + ph) * width + (j + pw);
              patches.push_back(image[idx * channel + c]);
            }
          }
        }
      }
    }

    // Allocate tensor for the patch data
    std::vector<int64_t> output_shape = {static_cast<int64_t>(patches.size()) / (patch_size * patch_size), patch_size * patch_size * channel};
    output.Allocate(output_shape);  // Allocate memory for the output tensor

    // Allocate a temporary non-const buffer for the output data
    std::vector<float> temp_buffer(patches.size());

    // Copy the data into the non-const buffer
    std::copy(patches.begin(), patches.end(), temp_buffer.begin());

    // Assuming output.Allocate returns a const pointer, we can't directly write to it.
    // So, we will copy the temp_buffer into the const data pointer.
    auto* p_output_image = const_cast<float*>(output.Data());  // Convert the const pointer to non-const pointer

    // Copy the data into the output tensor
    std::copy(temp_buffer.begin(), temp_buffer.end(), p_output_image);

    return {};
  }

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    // Needed for image transform structure
    return {};
  }
};

}  // namespace ort_extensions
