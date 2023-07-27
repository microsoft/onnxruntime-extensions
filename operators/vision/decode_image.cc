// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "decode_image.hpp"

#include <opencv2/imgcodecs.hpp>
#include "narrow.h"

namespace ort_extensions {

void KernelDecodeImage::Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) const {
  // Setup inputs
  const auto& dimensions = input.Shape();
  if (dimensions.size() != 1ULL) {
    ORTX_CXX_API_THROW("[DecodeImage]: Raw image bytes with 1D shape expected.", ORT_INVALID_ARGUMENT);
  }

  const int64_t encoded_image_data_len = input.NumberOfElement();

  // Decode the image
  const std::vector<int32_t> encoded_image_sizes{1, static_cast<int32_t>(encoded_image_data_len)};
  const void* encoded_image_data = input.Data();
  const cv::Mat encoded_image(encoded_image_sizes, CV_8UC1, const_cast<void*>(encoded_image_data));
  const cv::Mat decoded_image = cv::imdecode(encoded_image, cv::IMREAD_COLOR);

  if (decoded_image.data == nullptr) {
    ORTX_CXX_API_THROW("[DecodeImage] Invalid input. Failed to decode image.", ORT_INVALID_ARGUMENT);
  };

  // Setup output & copy to destination
  const cv::Size decoded_image_size = decoded_image.size();
  const int64_t height = decoded_image_size.height;
  const int64_t width = decoded_image_size.width;
  const int64_t colors = decoded_image.elemSize();  //  == 3 as it's BGR

  const std::vector<int64_t> output_dims{height, width, colors};
  uint8_t* decoded_image_data = output.Allocate(output_dims);
  memcpy(decoded_image_data, decoded_image.data, narrow<size_t>(height * width * colors));
}
}  // namespace ort_extensions
