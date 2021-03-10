// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_regex_split.hpp"
#include "string_regex_split_re.hpp"
#include <vector>
#include <cmath>
#include "string_common.h"

KernelStringRegexSplitWithOffsets::KernelStringRegexSplitWithOffsets(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api, info) {
}

void KernelStringRegexSplitWithOffsets::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* pattern = ort_.KernelContext_GetInput(context, 1);
  const OrtValue* keep_pattern = ort_.KernelContext_GetInput(context, 2);

  std::vector<std::string> str_input, str_pattern, str_keep_pattern;
  GetTensorMutableDataString(api_, ort_, context, input, str_input);
  GetTensorMutableDataString(api_, ort_, context, pattern, str_pattern);
  GetTensorMutableDataString(api_, ort_, context, keep_pattern, str_keep_pattern);

  // Verifications
  OrtTensorDimensions keep_pattern_dimensions(ort_, keep_pattern);
  if (str_pattern.size() != 1)
    throw std::runtime_error(MakeString(
        "pattern (second input) must contain only one element. It has ",
        str_pattern.size(), " values."));
  if (str_keep_pattern.size() > 1)
    throw std::runtime_error(MakeString(
        "Third input must contain only one element. It has ",
        str_keep_pattern.size(), " values."));
  if (str_pattern[0].empty()) {
    throw std::runtime_error("Splitting pattern cannot be empty.");
  }

  OrtTensorDimensions dimensions(ort_, input);
  re2::RE2 reg(str_pattern[0]);
  bool include_delimiter = (str_keep_pattern.size() == 1) && (!str_keep_pattern[0].empty());
  re2::RE2 keep_reg(include_delimiter ? str_keep_pattern[0] : "");
  std::vector<std::string> all_tokens;
  std::vector<int64_t> all_indices;
  std::vector<int64_t> row_indices;

  for (int64_t i = 0; i < dimensions[0]; i++) {
    row_indices.push_back(all_tokens.size());
    std::vector<std::string_view> tokens;
    std::vector<int64_t> begin_offsets;
    std::vector<int64_t> end_offsets;
    RegexSplitImpl(str_input[i], reg,
                   include_delimiter, keep_reg,
                   tokens, begin_offsets, end_offsets);
    all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
    for (size_t j = 0; j < begin_offsets.size(); ++j) {
      all_indices.push_back(i);
      all_indices.push_back(begin_offsets[j]);
      all_indices.push_back(end_offsets[j]);
    }
  }
  row_indices.push_back(all_tokens.size());

  // Setup output
  std::vector<int64_t> dim_out_text{(int64_t)all_tokens.size()};
  OrtValue* output_text = ort_.KernelContext_GetOutput(context, 0, dim_out_text.data(), dim_out_text.size());
  FillTensorDataString(api_, ort_, context, all_tokens, output_text);

  std::vector<int64_t> dim_out_indices{(int64_t)all_indices.size() / 3, 3};
  OrtValue* output_indices = ort_.KernelContext_GetOutput(context, 1, dim_out_indices.data(), dim_out_indices.size());
  int64_t* p_output_indices = ort_.GetTensorMutableData<int64_t>(output_indices);
  memcpy(p_output_indices, all_indices.data(), all_indices.size() * sizeof(int64_t));

  std::vector<int64_t> dim_out_rows{(int64_t)row_indices.size()};
  OrtValue* output_rows = ort_.KernelContext_GetOutput(context, 2, dim_out_rows.data(), dim_out_rows.size());
  p_output_indices = ort_.GetTensorMutableData<int64_t>(output_rows);
  memcpy(p_output_indices, row_indices.data(), row_indices.size() * sizeof(int64_t));
}

void* CustomOpStringRegexSplitWithOffsets::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelStringRegexSplitWithOffsets(api, info);
};

const char* CustomOpStringRegexSplitWithOffsets::GetName() const { return "StringRegexSplitWithOffsets"; };

size_t CustomOpStringRegexSplitWithOffsets::GetInputTypeCount() const {
  return 3;
};

ONNXTensorElementDataType CustomOpStringRegexSplitWithOffsets::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringRegexSplitWithOffsets::GetOutputTypeCount() const {
  return 3;
};

ONNXTensorElementDataType CustomOpStringRegexSplitWithOffsets::GetOutputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 1:
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      throw std::runtime_error("StringRegexSplitWithOffsets has 3 outputs.");
  }
};
