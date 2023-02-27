// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <regex>
#include "ocos.h"
#include "string_utils.h"

// See https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/regex_split_with_offsets.md.
struct KernelStringECMARegexSplitWithOffsets : BaseKernel {
  KernelStringECMARegexSplitWithOffsets(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context);

 private:
  bool ignore_case_;
};

struct CustomOpStringECMARegexSplitWithOffsets : OrtW::CustomOpBase<CustomOpStringECMARegexSplitWithOffsets, KernelStringECMARegexSplitWithOffsets> {
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

template <typename T>
void ECMARegexSplitImpl(const std::string& input, const std::regex& pattern,
                        bool include_delimiter, const std::regex& include_delim_regex,
                        std::vector<std::string_view>& tokens,
                        std::vector<T>& begin_offsets,
                        std::vector<T>& end_offsets) {
  size_t prev_pos = 0;
  for (auto it = std::sregex_iterator(input.begin(), input.end(), pattern); it != std::sregex_iterator(); it++) {
    int cur_pos = static_cast<int>(it->position());
    int matched_length = static_cast<int>(it->length());
    if (static_cast<decltype(it->position())>(prev_pos) != it->position()) {
      tokens.emplace_back(input.c_str() + prev_pos, cur_pos - prev_pos);
      begin_offsets.push_back(prev_pos);
      end_offsets.push_back(cur_pos);
      // update prev_pos for delimiter
      prev_pos = cur_pos;
    }

    if (include_delimiter && std::regex_match(it->str(), include_delim_regex)) {
      tokens.emplace_back(input.c_str() + prev_pos, matched_length);
      begin_offsets.push_back(prev_pos);
      end_offsets.push_back(prev_pos + matched_length);
    }

    // no mather include the delimiter, we should skip it
    prev_pos += matched_length;
  }

  if (prev_pos != input.length()) {
    tokens.emplace_back(input.c_str() + prev_pos, input.length() - prev_pos);
    begin_offsets.push_back(prev_pos);
    end_offsets.push_back(input.length());
  }
}
