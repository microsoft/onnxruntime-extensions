// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

// See https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/regex_split_with_offsets.md.
void KernelStringRegexSplitWithOffsets(const ortc::Tensor<std::string>& input,
                                       std::string_view str_pattern,
                                       std::string_view str_keep_pattern,
                                       ortc::Tensor<std::string>& output_text,
                                       ortc::Tensor<int64_t>& output_begin,
                                       ortc::Tensor<int64_t>& output_end,
                                       ortc::Tensor<int64_t>& output_offset);