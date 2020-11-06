// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_output.hpp"

KernelOne::KernelOne(OrtApi api) : BaseKernel(api) {
}

void KernelOne::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
  const float* X = ort_.GetTensorData<float>(input_X);
  const float* Y = ort_.GetTensorData<float>(input_Y);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
  for (int64_t i = 0; i < size; i++) {
    out[i] = X[i] + Y[i];
  }
}

void* CustomOpOne::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
  return new KernelOne(api);
};

const char* CustomOpOne::GetName() const {
  return "CustomOpOne";
};

size_t CustomOpOne::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpOne::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

size_t CustomOpOne::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpOne::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

KernelTwo::KernelTwo(OrtApi api) : BaseKernel(api) {
}

void KernelTwo::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const float* X = ort_.GetTensorData<float>(input_X);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  int32_t* out = ort_.GetTensorMutableData<int32_t>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
  for (int64_t i = 0; i < size; i++) {
    out[i] = (int32_t)(round(X[i]));
  }
}

void* CustomOpTwo::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
  return new KernelTwo(api);
};

const char* CustomOpTwo::GetName() const {
  return "CustomOpTwo";
};

size_t CustomOpTwo::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpTwo::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

size_t CustomOpTwo::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpTwo::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
};

KernelNegPos::KernelNegPos(OrtApi api) : BaseKernel(api) {
}

void KernelNegPos::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const float* X = ort_.GetTensorData<float>(input_X);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);

  OrtValue* output0 = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out0 = ort_.GetTensorMutableData<float>(output0);
  OrtValue* output1 = ort_.KernelContext_GetOutput(context, 1, dimensions.data(), dimensions.size());
  float* out1 = ort_.GetTensorMutableData<float>(output1);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output0);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
  for (int64_t i = 0; i < size; i++) {
    if (X[i] > 0) {
      out0[i] = 0;
      out1[i] = X[i];
    } else {
      out0[i] = X[i];
      out1[i] = 0;
    }
  }
}

void* CustomOpNegPos::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
  return new KernelNegPos(api);
};

const char* CustomOpNegPos::GetName() const {
  return "NegPos";
};

size_t CustomOpNegPos::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpNegPos::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

size_t CustomOpNegPos::GetOutputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpNegPos::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};
