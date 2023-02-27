// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <vector>
#include "ocos.h"
#include "ustring.h"
#include "string_utils.h"
#include "string_tensor.h"

struct KernelWordpieceTokenizer : BaseKernel {
  KernelWordpieceTokenizer(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context);

 private:
  int64_t max_input_chars_per_word_;
  std::u32string suffix_indicator_;
  ustring unk_token_;
  std::unordered_map<std::u32string, int32_t> vocab_;
};

struct CustomOpWordpieceTokenizer : OrtW::CustomOpBase<CustomOpWordpieceTokenizer, KernelWordpieceTokenizer> {
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

void KernelWordpieceTokenizer_Split(const std::u32string& suffix_indicator,
                                    const std::u32string& text,
                                    std::vector<std::u32string>& words);

void KernelWordpieceTokenizer_Tokenizer(const std::unordered_map<std::u32string, int32_t>& vocab,
                                        const std::u32string& suffix_indicator,
                                        const ustring& unk_token,
                                        const std::vector<ustring>& texts,
                                        std::vector<ustring>& tokens,
                                        std::vector<int32_t>& indices,
                                        std::vector<int64_t>& rows,
                                        const int64_t* existing_rows = nullptr,
                                        int64_t n_existing_rows = 0,
                                        int64_t max_input_chars_per_word = 200);
