// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_regex_split.hpp"
#include "string_regex_split_re.hpp"
#include "string_tensor.h"
#include <vector>
#include <cmath>

KernelStringRegexSplitWithOffsets::KernelStringRegexSplitWithOffsets(const OrtApi& api, const OrtKernelInfo& info)
    : BaseKernel(api, info) {
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
    ORTX_CXX_API_THROW(MakeString("pattern (second input) must contain only one element. It has ",
                                  str_pattern.size(), " values."),
                       ORT_INVALID_ARGUMENT);
  if (str_keep_pattern.size() > 1)
    ORTX_CXX_API_THROW(MakeString("Third input must contain only one element. It has ",
                                  str_keep_pattern.size(), " values."),
                       ORT_INVALID_ARGUMENT);
  if (str_pattern[0].empty())
    ORTX_CXX_API_THROW("Splitting pattern cannot be empty.", ORT_INVALID_ARGUMENT);

  OrtTensorDimensions dimensions(ort_, input);
  bool include_delimiter = (str_keep_pattern.size() == 1) && (!str_keep_pattern[0].empty());

  re2::RE2 reg(str_pattern[0]);
  re2::RE2 keep_reg(include_delimiter ? str_keep_pattern[0] : "");

  std::vector<std::string> all_tokens;
  std::vector<int64_t> all_begin_offsets, all_end_offsets;
  std::vector<int64_t> row_offsets;

  for (int64_t i = 0; i < dimensions[0]; i++) {
    row_offsets.push_back(all_begin_offsets.size());
    std::vector<std::string_view> tokens;
    std::vector<int64_t> begin_offsets;
    std::vector<int64_t> end_offsets;
    RegexSplitImpl(str_input[static_cast<size_t>(i)], reg,
                   include_delimiter, keep_reg,
                   tokens, begin_offsets, end_offsets);
    all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
    for (size_t j = 0; j < begin_offsets.size(); ++j) {
      all_begin_offsets.push_back(begin_offsets[j]);
      all_end_offsets.push_back(end_offsets[j]);
    }
  }
  row_offsets.push_back(all_begin_offsets.size());

  // Setup output
  std::vector<int64_t> dim_out{(int64_t)all_tokens.size()};
  OrtValue* output_text = ort_.KernelContext_GetOutput(context, 0, dim_out.data(), dim_out.size());
  FillTensorDataString(api_, ort_, context, all_tokens, output_text);

  OrtValue* output = ort_.KernelContext_GetOutput(context, 1, dim_out.data(), dim_out.size());
  int64_t* p_output = ort_.GetTensorMutableData<int64_t>(output);
  memcpy(p_output, all_begin_offsets.data(), all_begin_offsets.size() * sizeof(int64_t));

  output = ort_.KernelContext_GetOutput(context, 2, dim_out.data(), dim_out.size());
  p_output = ort_.GetTensorMutableData<int64_t>(output);
  memcpy(p_output, all_end_offsets.data(), all_end_offsets.size() * sizeof(int64_t));

  std::vector<int64_t> dim_out_row{(int64_t)row_offsets.size()};
  output = ort_.KernelContext_GetOutput(context, 3, dim_out_row.data(), dim_out_row.size());
  p_output = ort_.GetTensorMutableData<int64_t>(output);
  memcpy(p_output, row_offsets.data(), row_offsets.size() * sizeof(int64_t));
}

const char* CustomOpStringRegexSplitWithOffsets::GetName() const { return "StringRegexSplitWithOffsets"; };

size_t CustomOpStringRegexSplitWithOffsets::GetInputTypeCount() const {
  return 3;
};

ONNXTensorElementDataType CustomOpStringRegexSplitWithOffsets::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringRegexSplitWithOffsets::GetOutputTypeCount() const {
  return 4;
};

ONNXTensorElementDataType CustomOpStringRegexSplitWithOffsets::GetOutputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 1:
    case 2:
    case 3:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      ORTX_CXX_API_THROW(MakeString("StringRegexSplitWithOffsets has 4 outputs but index is ", index, "."),
                         ORT_INVALID_ARGUMENT);
  }
};
