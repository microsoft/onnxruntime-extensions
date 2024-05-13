// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ocos.h"
#include "ortx_common.h"

#include <cstdint>

OrtxStatus image_decoder(const ortc::Tensor<uint8_t>& input,
                         ortc::Tensor<uint8_t>& output) {
  auto& dimensions = input.Shape();
  if (dimensions.size() != 1ULL) {
    return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Only raw image formats are supported."};
  }

  // Get data & the length
  const uint8_t* const encoded_image_data = input.Data();
  const int64_t encoded_image_data_len = input.NumberOfElement();

  // Decode the image
  const std::vector<int32_t> encoded_image_sizes{1, static_cast<int32_t>(encoded_image_data_len)};
  const cv::Mat encoded_image(encoded_image_sizes, CV_8UC1,
                              const_cast<void*>(static_cast<const void*>(encoded_image_data)));
  const cv::Mat decoded_image = cv::imdecode(encoded_image, cv::IMREAD_COLOR);

  // Setup output & copy to destination
  const cv::Size decoded_image_size = decoded_image.size();
  const int64_t colors = 3;

  const std::vector<int64_t> output_dimensions{decoded_image_size.height, decoded_image_size.width, colors};
  uint8_t* const decoded_image_data = output.Allocate(output_dimensions);
  memcpy(decoded_image_data, decoded_image.data, decoded_image.total() * decoded_image.elemSize());

  return {};
}
