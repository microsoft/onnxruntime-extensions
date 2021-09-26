// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_contains.hpp"
#include "string_tensor.h"
#include <vector>

KernelStringContains::KernelStringContains(OrtApi api) : BaseKernel(api) {
}

void KernelStringContains::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* text = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* substr = ort_.KernelContext_GetInput(context, 1);
  OrtTensorDimensions text_dim(ort_, text);
  OrtTensorDimensions substr_dim(ort_, substr);

  if (text_dim != substr_dim)
    ORT_CXX_API_THROW("Dimension mismatch", ORT_INVALID_ARGUMENT);

  int64_t batch_size = text_dim[0];

  OrtTensorDimensions output_dim;
  output_dim.emplace_back(text_dim[0]);

  std::vector<std::string> text_value;
  std::vector<std::string> substr_value;
  GetTensorMutableDataString(api_, ort_, context, text, text_value);
  GetTensorMutableDataString(api_, ort_, context, substr, substr_value);

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dim.data(), output_dim.size());
  auto* output_data = ort_.GetTensorMutableData<bool>(output);

  for (size_t i = 0; i < text_value.size(); i++) {
    if (text_value[i].find(substr_value[i]) != std::string::npos) {
      output_data[i] = true;
    } else {
      output_data[i] = false;
    }
  }
}

void* CustomOpStringContains::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelStringContains(api);
};

const char* CustomOpStringContains::GetName() const { return "StringContains"; };

size_t CustomOpStringContains::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpStringContains::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringContains::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringContains::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
};