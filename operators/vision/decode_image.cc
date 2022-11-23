// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "decode_image.hpp"

#include <opencv2/imgcodecs.hpp>

namespace ort_extensions {

void KernelDecodeImage::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* const inputs = ort_.KernelContext_GetInput(context, 0ULL);
  OrtTensorDimensions dimensions(ort_, inputs);
  if (dimensions.size() != 1ULL) {
    ORT_CXX_API_THROW("[DecodeImage]: Raw image bytes with 1D shape expected.", ORT_INVALID_ARGUMENT);
  }

  OrtTensorTypeAndShapeInfo* input_info = ort_.GetTensorTypeAndShape(inputs);
  const int64_t encoded_image_data_len = ort_.GetTensorShapeElementCount(input_info);
  ort_.ReleaseTensorTypeAndShapeInfo(input_info);

  // Decode the image
  const std::vector<int32_t> encoded_image_sizes{1, static_cast<int32_t>(encoded_image_data_len)};
  const void* encoded_image_data = ort_.GetTensorData<uint8_t>(inputs);  // uint8 data
  const cv::Mat encoded_image(encoded_image_sizes, CV_8UC1, const_cast<void*>(encoded_image_data));
  const cv::Mat decoded_image = cv::imdecode(encoded_image, cv::IMREAD_COLOR);

  if (decoded_image.data == nullptr) {
    ORT_CXX_API_THROW("[DecodeImage] Invalid input. Failed to decode image.", ORT_INVALID_ARGUMENT);
  };

  // Setup output & copy to destination
  const cv::Size decoded_image_size = decoded_image.size();
  const int64_t colors = decoded_image.elemSize();  //  == 3 as it's BGR

  const std::vector<int64_t> output_dims{decoded_image_size.height, decoded_image_size.width, colors};
  OrtValue* output_value = ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());
  uint8_t* decoded_image_data = ort_.GetTensorMutableData<uint8_t>(output_value);
  memcpy(decoded_image_data, decoded_image.data, decoded_image_size.height * decoded_image_size.width * colors);
}
}  // namespace ort_extensions
