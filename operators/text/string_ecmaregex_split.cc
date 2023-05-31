// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <algorithm>
#include <regex>
#include <vector>
#include <cmath>
#include "string_ecmaregex_split.hpp"
#include "string_tensor.h"

KernelStringECMARegexSplitWithOffsets::KernelStringECMARegexSplitWithOffsets(const OrtApi& api,
                                                                             const OrtKernelInfo& info)
    : BaseKernel(api, info) {
  ignore_case_ = TryToGetAttributeWithDefault("ignore_case", false);
}

void KernelStringECMARegexSplitWithOffsets::Compute(const ortc::Tensor<std::string>& input,
                                                    std::string_view pattern,
                                                    std::string_view keep_pattern,
                                                    ortc::Tensor<std::string>& output_text,
                                                    ortc::Tensor<int64_t>& output1,
                                                    ortc::Tensor<int64_t>& output2,
                                                    ortc::Tensor<int64_t>& output3) {
  // Setup inputs
  auto& str_input = input.Data();

  auto& dimensions = input.Shape();
  bool include_delimiter = !keep_pattern.empty();

  auto regex_flag = std::regex_constants::ECMAScript;
  if (ignore_case_) {
    regex_flag |= std::regex_constants::icase;
  }

  std::regex reg(pattern.data(), regex_flag);
  std::regex keep_reg(include_delimiter ? keep_pattern.data() : "", regex_flag);

  std::vector<std::string> all_tokens;
  std::vector<int64_t> all_begin_offsets, all_end_offsets;
  std::vector<int64_t> row_offsets;

  for (int64_t i = 0; i < dimensions[0]; i++) {
    row_offsets.push_back(all_begin_offsets.size());
    std::vector<std::string_view> tokens;
    std::vector<int64_t> begin_offsets;
    std::vector<int64_t> end_offsets;
    ECMARegexSplitImpl(str_input[static_cast<size_t>(i)], reg,
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
  output_text.SetStringOutput(all_tokens, dim_out);

  int64_t* p_output = output1.Allocate(dim_out);
  memcpy(p_output, all_begin_offsets.data(), all_begin_offsets.size() * sizeof(int64_t));

  p_output = output2.Allocate(dim_out);
  memcpy(p_output, all_end_offsets.data(), all_end_offsets.size() * sizeof(int64_t));

  std::vector<int64_t> dim_out_row{(int64_t)row_offsets.size()};
  p_output = output3.Allocate(dim_out_row);
  memcpy(p_output, row_offsets.data(), row_offsets.size() * sizeof(int64_t));
}
