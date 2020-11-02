// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_upper.hpp"

KernelStringUpper::KernelStringUpper(OrtApi api) : BaseKernel(api) {
}

void KernelStringUpper::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const std::string* X = ort_.GetTensorData<std::string>(input_X);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  std::string* out = ort_.GetTensorMutableData<std::string>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
  for (int64_t i = 0; i < size; i++) {
    out[i] = X[i];
    std::transform(out[i].begin(), out[i].end(), out[i].begin(), ::toupper);
  }
}

void* CustomOpStringUpper::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
  return new KernelStringUpper(api);
};

const char* CustomOpStringUpper::GetName() const { return "StringUpper"; };

size_t CustomOpStringUpper::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringUpper::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringUpper::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringUpper::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

