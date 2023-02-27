// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_mapping.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>

KernelStringMapping::KernelStringMapping(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  std::string map = ort_.KernelInfoGetAttribute<std::string>(&info, "map");
  auto lines = SplitString(map, "\n", true);
  for (const auto& line : lines) {
    auto items = SplitString(line, "\t", true);

    if (items.size() != 2) {
      ORTX_CXX_API_THROW(std::string("[StringMapping]: Should only exist two items in one line, find error in line: ") + std::string(line), ORT_INVALID_GRAPH);
    }
    map_[std::string(items[0])] = std::string(items[1]);
  }
}

void KernelStringMapping::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> input_data;
  GetTensorMutableDataString(api_, ort_, context, input, input_data);

  OrtTensorDimensions dimensions(ort_, input);

  for (auto& str : input_data) {
    if (map_.find(str) != map_.end()) {
      str = map_[str];
    }
  }

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());

  FillTensorDataString(api_, ort_, context, input_data, output);
}

const char* CustomOpStringMapping::GetName() const { return "StringMapping"; };

size_t CustomOpStringMapping::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringMapping::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringMapping::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringMapping::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
