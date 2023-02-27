// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "ocos.h"
#include "string_utils.h"

#include <cstdint>

struct KernelImageDecoder : BaseKernel {
  KernelImageDecoder(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {}

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* const inputs = ort_.KernelContext_GetInput(context, 0ULL);
    OrtTensorDimensions dimensions(ort_, inputs);
    if (dimensions.size() != 1ULL) {
      ORTX_CXX_API_THROW("[ImageDecoder]: Only raw image formats are supported.", ORT_INVALID_ARGUMENT);
    }

    // Get data & the length
    const uint8_t* const encoded_image_data = ort_.GetTensorData<uint8_t>(inputs);

    OrtTensorTypeAndShapeInfo* const input_info = ort_.GetTensorTypeAndShape(inputs);
    const int64_t encoded_image_data_len = ort_.GetTensorShapeElementCount(input_info);
    ort_.ReleaseTensorTypeAndShapeInfo(input_info);

    // Decode the image
    const std::vector<int32_t> encoded_image_sizes{1, static_cast<int32_t>(encoded_image_data_len)};
    const cv::Mat encoded_image(encoded_image_sizes, CV_8UC1,
                                const_cast<void*>(static_cast<const void*>(encoded_image_data)));
    const cv::Mat decoded_image = cv::imdecode(encoded_image, cv::IMREAD_COLOR);

    // Setup output & copy to destination
    const cv::Size decoded_image_size = decoded_image.size();
    const int64_t colors = 3;

    const std::vector<int64_t> output_dimensions{decoded_image_size.height, decoded_image_size.width, colors};
    OrtValue* const output_value = ort_.KernelContext_GetOutput(
        context, 0, output_dimensions.data(), output_dimensions.size());
    uint8_t* const decoded_image_data = ort_.GetTensorMutableData<uint8_t>(output_value);
    memcpy(decoded_image_data, decoded_image.data, decoded_image.total() * decoded_image.elemSize());
  }
};

struct CustomOpImageDecoder : OrtW::CustomOpBase<CustomOpImageDecoder, KernelImageDecoder> {
  const char* GetName() const {
    return "ImageDecoder";
  }

  size_t GetInputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    switch (index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      default:
        ORTX_CXX_API_THROW(MakeString("Unexpected input index ", index), ORT_INVALID_ARGUMENT);
    }
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    switch (index) {
      case 0:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      default:
        ORTX_CXX_API_THROW(MakeString("Unexpected output index ", index), ORT_INVALID_ARGUMENT);
    }
  }
};
