// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <vector>
#include "ocos.h"
#include "ustring.h"
#include "string_utils.h"
#include "string_tensor.h"
#include "basic_tokenizer.hpp"

class WordPieceTokenizer {
  WordPieceTokenizer(std::string vocab, bool do_lower_case, bool do_basic_tokenize,
                     ustring unk_token, ustring sep_token, ustring pad_token, ustring  cls_token,
                     ustring mask_token, bool tokenize_chinese_chars, ustring suffix_indicator,
                     int64_t max_input_chars_per_word);
  std::vector<int64_t> Tokenize(std::string);
 private:
  std::unordered_map<ustring, int32_t> vocab_;
  bool do_basic_tokenize;
  std::shared_ptr<BasicTokenizer> basic_tokenizer_;
  bool do_lower_case_;
  ustring unk_token_;
  ustring sep_token_;
  ustring pad_token_;
  ustring cls_token_;
  ustring mask_token_;
  ustring suffix_indicator_;
  int64_t max_input_chars_per_word_;
};

struct KernelWordPieceTokenizer : BaseKernel {
  KernelWordPieceTokenizer(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);
 private:
  std::shared_ptr<WordPieceTokenizer> tokenizer_;
};

struct CustomOpWordPieceTokenizer : Ort::CustomOpBase<CustomOpWordPieceTokenizer, KernelWordPieceTokenizer> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
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
