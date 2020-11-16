// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_split.hpp"

KernelStringSplit::KernelStringSplit(OrtApi api) : BaseKernel(api) {
}

void KernelStringSplit::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const std::string* X = ort_.GetTensorData<std::string>(input_X);
  const OrtValue* input_sep = ort_.KernelContext_GetInput(context, 1);
  const std::string* sep = ort_.GetTensorData<std::string>(input_sep);
  const OrtValue* input_skip_empty = ort_.KernelContext_GetInput(context, 2);
  const bool* skip_empty = ort_.GetTensorData<bool>(input_skip_empty);

  // Setup output
  OrtTensorDimensions dimensions_sep(ort_, input_sep);
  if (dimensions_sep.size() != 1 || dimensions_sep[0] != 1)
    throw std::runtime_error("Input 2 is the delimiter, it has 1 element.");
  OrtTensorDimensions dimensions_skip_empty(ort_, input_skip_empty);
  if (dimensions_skip_empty.size() != 1 || dimensions_skip_empty[0] != 1)
    throw std::runtime_error("Input 3 is skip_empty, it has 1 element.");
  OrtTensorDimensions dimensions(ort_, input_X);
  if (dimensions.size() != 1)
    throw std::runtime_error("Only 1D tensor are supported as input.");

  std::vector<std::string> words;
  std::vector<int64_t> indices;
  int64_t maxc = 0;
  int64_t col;
  std::string delimiter = *sep;
  bool keep = !(*skip_empty);
  std::size_t current, previous = 0;
  for (int64_t row = 0; row < dimensions[0]; ++row) {
    const std::string& str = X[row];
    if (str.empty())
      continue;
    previous = 0;
    col = 0;
    current = str.find_first_of(delimiter);
    while (current != std::string::npos) {
      if (keep || current > previous) {
        words.push_back(str.substr(previous, current - previous));
        indices.push_back(row);
        indices.push_back(col);
        ++col;
      }
      previous = current + 1;
      current = str.find_first_of(delimiter, previous);
    }
    if (keep || current > previous) {
      words.push_back(str.substr(previous, current - previous));
      indices.push_back(row);
      indices.push_back(col);
      ++col;
    }
    maxc = col > maxc ? col : maxc;
  }

  std::vector<int64_t> shape_indices = {static_cast<int64_t>(indices.size()) / 2, 2};
  OrtValue* out_indices = ort_.KernelContext_GetOutput(context, 0, shape_indices.data(), shape_indices.size());

  std::vector<int64_t> shape_text(1, words.size());
  OrtValue* out_text = ort_.KernelContext_GetOutput(context, 1, shape_text.data(), shape_text.size());

  std::vector<int64_t> shape_shape(1, 2);
  OrtValue* out_shape = ort_.KernelContext_GetOutput(context, 2, shape_shape.data(), shape_shape.size());

  int64_t* p_indices = ort_.GetTensorMutableData<int64_t>(out_indices);
  std::string* p_text = ort_.GetTensorMutableData<std::string>(out_text);
  int64_t* p_shape = ort_.GetTensorMutableData<int64_t>(out_shape);

  memcpy(p_indices, indices.data(), indices.size() * sizeof(int64_t));
  p_shape[0] = dimensions[0];
  p_shape[1] = maxc;
  std::copy(words.begin(), words.end(), p_text);
}

void* CustomOpStringSplit::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
  return new KernelStringSplit(api);
};

const char* CustomOpStringSplit::GetName() const {
  return "StringSplit";
};

size_t CustomOpStringSplit::GetInputTypeCount() const {
  return 3;
};

ONNXTensorElementDataType CustomOpStringSplit::GetInputType(size_t index) const {
  switch (index) {
    case 0:
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:
      throw std::runtime_error(MakeString("Unexpected input index ", index));
  }
};

size_t CustomOpStringSplit::GetOutputTypeCount() const {
  return 3;
};

ONNXTensorElementDataType CustomOpStringSplit::GetOutputType(size_t index) const {
  switch (index) {
    case 0:
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    default:
      throw std::runtime_error(MakeString("Unexpected output index ", index));
  }
};
