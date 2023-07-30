// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "encode_image.hpp"

#include <opencv2/imgcodecs.hpp>

namespace ort_extensions {

void KernelEncodeImage::Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) const {
  // Setup inputs
  const auto dimensions_bgr = input.Shape();

  if (dimensions_bgr.size() != 3 || dimensions_bgr[2] != 3) {
    // expect {H, W, C} as that's the inverse of what decode_image produces.
    // we have no way to check if it's BGR or RGB though
    ORTX_CXX_API_THROW("[EncodeImage] requires rank 3 BGR input in channels last format.", ORT_INVALID_ARGUMENT);
  }

  // Get data & the length
  std::vector<int32_t> height_x_width{static_cast<int32_t>(dimensions_bgr[0]),   // H
                                      static_cast<int32_t>(dimensions_bgr[1])};  // W

  // data is const uint8_t but opencv2 wants void*.
  const void* bgr_data = input.Data();
  const cv::Mat bgr_image(height_x_width, CV_8UC3, const_cast<void*>(bgr_data));

  // don't know output size ahead of time so need to encode and then copy to output
  std::vector<uint8_t> encoded_image;
  if (!cv::imencode(extension_, bgr_image, encoded_image)) {
    ORTX_CXX_API_THROW("[EncodeImage] Image encoding failed.", ORT_INVALID_ARGUMENT);
  }

  // Setup output & copy to destination
  std::vector<int64_t> output_dimensions{static_cast<int64_t>(encoded_image.size())};
  uint8_t* data = output.Allocate(output_dimensions);
  memcpy(data, encoded_image.data(), encoded_image.size());
}

}  // namespace ort_extensions
