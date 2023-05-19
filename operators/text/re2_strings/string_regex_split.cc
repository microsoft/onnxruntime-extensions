// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_regex_split.hpp"
#include "string_regex_split_re.hpp"
#include "string_tensor.h"
#include <vector>
#include <cmath>

void KernelStringRegexSplitWithOffsets(const ortc::Tensor<std::string>& input,
                                       std::string_view str_pattern,
                                       const ortc::Tensor<std::string>& str_keep_pattern,
                                       ortc::Tensor<std::string>& output_text,
                                       ortc::Tensor<int64_t>& output_begin,
                                       ortc::Tensor<int64_t>& output_end,
                                       ortc::Tensor<int64_t>& output_offset) {
  // Setup inputs

  std::vector<std::string> str_input(input.Data());

  if (str_pattern.empty()) {
    ORTX_CXX_API_THROW("Splitting pattern cannot be empty.", ORT_INVALID_ARGUMENT);
  }
  if (str_keep_pattern.Data().size() > 1) {
    ORTX_CXX_API_THROW(MakeString("Third input must contain only one element. It has ",
                                  str_keep_pattern.Data().size(), " values."),
                       ORT_INVALID_ARGUMENT);
  }
  auto dimensions = input.Shape();
  bool include_delimiter = (str_keep_pattern.Data().size() == 1) && (!str_keep_pattern.Data()[0].empty());

  re2::RE2 reg(str_pattern.data());
  re2::RE2 keep_reg(include_delimiter ? str_keep_pattern.Data()[0].data() : "");

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
  output_text.SetStringOutput(all_tokens, dim_out);

  auto output_raw = output_begin.Allocate(dim_out);
  memcpy(output_raw, all_begin_offsets.data(), all_begin_offsets.size() * sizeof(int64_t));

  output_raw = output_end.Allocate(dim_out);
  memcpy(output_raw, all_end_offsets.data(), all_end_offsets.size() * sizeof(int64_t));

  std::vector<int64_t> dim_out_row{(int64_t)row_offsets.size()};
  output_raw = output_offset.Allocate(dim_out_row);
  memcpy(output_raw, row_offsets.data(), row_offsets.size() * sizeof(int64_t));
}
