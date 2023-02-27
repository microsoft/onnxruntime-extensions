// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "masked_fill.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>

KernelMaskedFill::KernelMaskedFill(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
}

void KernelMaskedFill::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_value = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_mask = ort_.KernelContext_GetInput(context, 1);

  OrtTensorDimensions value_dimensions(ort_, input_value);
  OrtTensorDimensions mask_dimensions(ort_, input_mask);

  if (!(value_dimensions.IsScalar() || value_dimensions.IsVector())) {
    ORTX_CXX_API_THROW("[MaskedFill]: the dimension of input value should be vector or scalar.", ORT_INVALID_ARGUMENT);
  }

  if (value_dimensions != mask_dimensions) {
    ORTX_CXX_API_THROW("[MaskedFill]: the dimension of input value and mask should be same.", ORT_INVALID_ARGUMENT);
  }

  std::vector<std::string> value;
  const bool* mask = nullptr;

  GetTensorMutableDataString(api_, ort_, context, input_value, value);
  mask = ort_.GetTensorData<bool>(input_mask);

  std::vector<std::string> result;
  std::vector<int64_t> result_dimension;

  for (size_t i = 0; i < value.size(); i++) {
    if (!mask[i]) {
      continue;
    }

    result.push_back(value[i]);
  }
  result_dimension.push_back(result.size());

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, result_dimension.data(), result_dimension.size());

  FillTensorDataString(api_, ort_, context, result, output);
}

const char* CustomOpMaskedFill::GetName() const { return "MaskedFill"; };

size_t CustomOpMaskedFill::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpMaskedFill::GetInputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:
      ORTX_CXX_API_THROW(MakeString("Unexpected input index ", index), ORT_INVALID_ARGUMENT);
  }
};

size_t CustomOpMaskedFill::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpMaskedFill::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
