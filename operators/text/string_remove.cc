// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_remove.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>


KernelStringRemove::KernelStringRemove(OrtApi api, const OrtKernelInfo* /*info*/) : BaseKernel(api) {
}

void KernelStringRemove::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_string = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_condition = ort_.KernelContext_GetInput(context, 1);

  OrtTensorDimensions string_dimensions(ort_, input_string);
  OrtTensorDimensions condition_dimensions(ort_, input_condition);

  if ((!(string_dimensions.IsScalar() || string_dimensions.IsVector())) ||
      (!(condition_dimensions.IsVector() || condition_dimensions.IsVector()))) {
    ORT_CXX_API_THROW("[StringMapping]: the dimension of input string and condition should be [n] or [1, n] vector.", ORT_INVALID_ARGUMENT);
  }

  if (string_dimensions.Size() != condition_dimensions.Size()) {
    ORT_CXX_API_THROW("[StringMapping]: the dimension of input string and condition should be same.", ORT_INVALID_ARGUMENT);
  }

  std::vector<std::string> strs;
  const int64_t * conditions = nullptr;

  GetTensorMutableDataString(api_, ort_, context, input_string, strs);
  conditions = ort_.GetTensorData<int64_t>(input_condition);

  std::vector<std::string> result;
  std::vector<int64_t> result_dimension;

  for (int i = 0; i < strs.size(); i++) {
    if (conditions[i] == 0) {
      continue;
    }

    result.push_back(strs[i]);
  }
  result_dimension.push_back(result.size());

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, result_dimension.data(), result_dimension.size());

  FillTensorDataString(api_, ort_, context, result, output);
}

void* CustomOpStringRemove::CreateKernel(OrtApi api, const OrtKernelInfo*  info) const {
  return new KernelStringRemove(api, info);
};

const char* CustomOpStringRemove::GetName() const { return "StringRemove"; };

size_t CustomOpStringRemove::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpStringRemove::GetInputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      ORT_CXX_API_THROW(MakeString("Unexpected input index ", index), ORT_INVALID_ARGUMENT);
  }};

size_t CustomOpStringRemove::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringRemove::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
