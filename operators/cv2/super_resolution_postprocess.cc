// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "super_resolution_postprocess.hpp"
#include "string_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdint>

KernelSuperResolutionPostProcess::KernelSuperResolutionPostProcess(const OrtApi& api) : BaseKernel(api) {}

void KernelSuperResolutionPostProcess::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* const input_y = ort_.KernelContext_GetInput(context, 0ULL);
  const OrtValue* const input_cr = ort_.KernelContext_GetInput(context, 1ULL);
  const OrtValue* const input_cb = ort_.KernelContext_GetInput(context, 2ULL);

  const OrtTensorDimensions dimensions_y(ort_, input_y);
  const OrtTensorDimensions dimensions_cr(ort_, input_cr);
  const OrtTensorDimensions dimensions_cb(ort_, input_cb);
  if ((dimensions_y.size() != 4ULL) || (dimensions_cr.size() != 4ULL) || (dimensions_cb.size() != 4ULL)) {
    throw std::runtime_error("Expecting 3 channels y, cr, and cb.");
  }

  // Get data & the length
  const float* const channel_y_data = ort_.GetTensorData<float>(input_y);
  const float* const channel_cr_data = ort_.GetTensorData<float>(input_cr);
  const float* const channel_cb_data = ort_.GetTensorData<float>(input_cb);

  cv::Mat y(
    std::vector<int32_t>{static_cast<int32_t>(dimensions_y[2]), static_cast<int32_t>(dimensions_y[3])},
    CV_32F, const_cast<void*>(static_cast<const void*>(channel_y_data)));
  cv::Mat cr(
    std::vector<int32_t>{static_cast<int32_t>(dimensions_cr[2]), static_cast<int32_t>(dimensions_cr[3])},
    CV_32F, const_cast<void*>(static_cast<const void*>(channel_cr_data)));
  cv::Mat cb(
    std::vector<int32_t>{static_cast<int32_t>(dimensions_cb[2]), static_cast<int32_t>(dimensions_cb[3])},
    CV_32F, const_cast<void*>(static_cast<const void*>(channel_cb_data)));

  // Scale the individual channels
  y *= 255.0;
  cv::resize(cr, cr, y.size(), 0, 0, cv::INTER_CUBIC);
  cv::resize(cb, cb, y.size(), 0, 0, cv::INTER_CUBIC);

  // Merge the channels
  const cv::Mat channels[] = {y, cr, cb};
  cv::Mat ycrcb_image;
  cv::merge(channels, 3, ycrcb_image);

  // Convert it back to BGR format
  cv::Mat bgr_image;
  cv::cvtColor(ycrcb_image, bgr_image, cv::COLOR_YCrCb2BGR);

  // Encode it as jpg
  std::vector<uchar> encoded_image;
  cv::imencode(".jpg", bgr_image, encoded_image);

  // Setup output & copy to destination
  const std::vector<int64_t> output_dimensions{1LL, static_cast<int64_t>(encoded_image.size())};
  OrtValue* const output_value = ort_.KernelContext_GetOutput(
      context, 0, output_dimensions.data(), output_dimensions.size());
  float* const data = ort_.GetTensorMutableData<float>(output_value);
  memcpy(data, encoded_image.data(), encoded_image.size());
}

const char* CustomOpSuperResolutionPostProcess::GetName() const {
  return "SuperResolutionPostProcess";
}

size_t CustomOpSuperResolutionPostProcess::GetInputTypeCount() const {
  return 3;
}

ONNXTensorElementDataType CustomOpSuperResolutionPostProcess::GetInputType(size_t index) const {
  switch (index) {
    case 0:
    case 1:
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    default:
      ORT_CXX_API_THROW(MakeString("Unexpected input index ", index), ORT_INVALID_ARGUMENT);
  }
}

size_t CustomOpSuperResolutionPostProcess::GetOutputTypeCount() const {
  return 1;
}

ONNXTensorElementDataType CustomOpSuperResolutionPostProcess::GetOutputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    default:
      ORT_CXX_API_THROW(MakeString("Unexpected output index ", index), ORT_INVALID_ARGUMENT);
  }
}
