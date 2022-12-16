// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_OPENCV_CODECS

#include "super_resolution_preprocess.hpp"
#include "string_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdint>

KernelSuperResolutionPreProcess::KernelSuperResolutionPreProcess(const OrtApi& api) : BaseKernel(api) {}

void KernelSuperResolutionPreProcess::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* const inputs = ort_.KernelContext_GetInput(context, 0ULL);
  OrtTensorDimensions dimensions(ort_, inputs);
  if (dimensions.size() != 1ULL) {
    throw std::runtime_error("Only raw image formats are supported.");
  }

  // Get data & the length
  const uint8_t* const encoded_bgr_image_data = ort_.GetTensorData<uint8_t>(inputs);

  OrtTensorTypeAndShapeInfo* const input_info = ort_.GetTensorTypeAndShape(inputs);
  const int64_t encoded_bgr_image_data_len = ort_.GetTensorShapeElementCount(input_info);
  ort_.ReleaseTensorTypeAndShapeInfo(input_info);

  // Decode the image
  const std::vector<int32_t> encoded_bgr_image_sizes{1, static_cast<int32_t>(encoded_bgr_image_data_len)};
  const cv::Mat encoded_bgr_image(encoded_bgr_image_sizes, CV_8UC1,
                                  const_cast<void*>(static_cast<const void*>(encoded_bgr_image_data)));
  // OpenCV decodes images in BGR format.
  // Ref: https://stackoverflow.com/a/44359400
  const cv::Mat decoded_bgr_image = cv::imdecode(encoded_bgr_image, cv::IMREAD_COLOR);

  cv::Mat normalized_bgr_image;
  decoded_bgr_image.convertTo(normalized_bgr_image, CV_32F);

  cv::Mat ycrcb_image;
  cv::cvtColor(normalized_bgr_image, ycrcb_image, cv::COLOR_BGR2YCrCb);

  cv::Mat channels[3];
  cv::split(ycrcb_image, channels);
  channels[0] /= 255.0;

  // Setup output & copy to destination
  for (int32_t i = 0; i < 3; ++i) {
    const cv::Mat& channel = channels[i];
    const cv::Size size = channel.size();

    const std::vector<int64_t> output_dimensions{1LL, 1LL, size.height, size.width};
    OrtValue* const output_value = ort_.KernelContext_GetOutput(
        context, i, output_dimensions.data(), output_dimensions.size());
    float* const data = ort_.GetTensorMutableData<float>(output_value);
    memcpy(data, channel.data, channel.total() * channel.elemSize());
  }
}

const char* CustomOpSuperResolutionPreProcess::GetName() const {
  return "SuperResolutionPreProcess";
}

size_t CustomOpSuperResolutionPreProcess::GetInputTypeCount() const {
  return 1;
}

ONNXTensorElementDataType CustomOpSuperResolutionPreProcess::GetInputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    default:
      ORTX_CXX_API_THROW(MakeString("Unexpected input index ", index), ORT_INVALID_ARGUMENT);
  }
}

size_t CustomOpSuperResolutionPreProcess::GetOutputTypeCount() const {
  return 3;
}

ONNXTensorElementDataType CustomOpSuperResolutionPreProcess::GetOutputType(size_t index) const {
  switch (index) {
    case 0:
    case 1:
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    default:
      ORTX_CXX_API_THROW(MakeString("Unexpected output index ", index), ORT_INVALID_ARGUMENT);
  }
}

#endif  // ENABLE_OPENCV_CODECS
