// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "string_kernels.hpp"

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

KernelStringJoin::KernelStringJoin(OrtApi api) : BaseKernel(api) {
}

void KernelStringJoin::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const std::string* X = ort_.GetTensorData<std::string>(input_X);
  const OrtValue* input_sep = ort_.KernelContext_GetInput(context, 1);
  const std::string* sep = ort_.GetTensorData<std::string>(input_sep);

  // Setup output
  OrtTensorDimensions dimensions_sep(ort_, input_sep);
  if (dimensions_sep.size() != 1 || dimensions_sep[0] != 1)
    throw std::runtime_error("Input 2 is the separator, it has 1 element.");
  OrtTensorDimensions dimensions(ort_, input_X);
  if (dimensions.size() != 2)
    throw std::runtime_error(MakeString("Input 1 must have 2 dimensions but has ", dimensions.size(), "."));
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), 1);
  std::string* out = ort_.GetTensorMutableData<std::string>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
  int64_t index = 0;
  for (int64_t i = 0; i < size; ++i) {
    std::ostringstream st;
    for (int64_t j = 0; j < dimensions[1] - 1; ++j, ++index) {
      st << X[index] << *sep;
    }
    st << X[index++];
    out[i] = st.str();
  }
}

void* CustomOpStringJoin::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
  return new KernelStringJoin(api);
};

const char* CustomOpStringJoin::GetName() const {
  return "StringJoin";
};

size_t CustomOpStringJoin::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpStringJoin::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringJoin::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringJoin::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
