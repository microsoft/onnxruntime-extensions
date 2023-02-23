// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_length.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>


KernelStringLength::KernelStringLength(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
}

void KernelStringLength::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> input_data;
  GetTensorMutableDataString(api_, ort_, context, input, input_data);

  OrtTensorDimensions dimensions(ort_, input);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  auto* output_data = ort_.GetTensorMutableData<int64_t>(output);

  for (int i = 0; i < dimensions.Size(); i++) {
    output_data[i] = ustring(input_data[i]).size();
  }
}

const char* CustomOpStringLength::GetName() const { return "StringLength"; };

size_t CustomOpStringLength::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringLength::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringLength::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringLength::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};
