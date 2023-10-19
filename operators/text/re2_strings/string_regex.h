// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

// See https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/regex_split_with_offsets.md.
OrtStatusPtr KernelStringRegexSplitWithOffsets(const ortc::Tensor<std::string>& input,
                                               std::string_view str_pattern,
                                               const ortc::Tensor<std::string>& str_keep_pattern,
                                               ortc::Tensor<std::string>& output_text,
                                               ortc::Tensor<int64_t>& output_begin,
                                               ortc::Tensor<int64_t>& output_end,
                                               ortc::Tensor<int64_t>& output_offset);

struct KernelStringRegexReplace {
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    return OrtW::GetOpAttribute(info, "global_replace", global_replace_);
  }

  OrtStatusPtr Compute(const ortc::Tensor<std::string>& input,
                       std::string_view str_pattern,
                       std::string_view str_rewrite,
                       ortc::Tensor<std::string>& output) const;

 protected:
  int64_t global_replace_{1};
};
