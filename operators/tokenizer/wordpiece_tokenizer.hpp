// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <vector>
#include "ocos.h"
#include "ustring.h"
#include "string_utils.h"
#include "string_tensor.h"

class WordPieceTokenizer{
 public:
  WordPieceTokenizer(std::string vocab, ustring unk_token, int max_input_chars_per_word = 100);
  std::vector<ustring> Tokenizer(ustring text);
  std::vector<int64_t> Encode(std::vector<ustring> token);
 private:
  int64_t max_input_chars_per_word_;
  ustring suffix_indicator_;
  ustring unk_token_;
  std::unordered_map<ustring, int32_t> vocab_;
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

void KernelWordPieceTokenizer_Split(const ustring& suffix_indicator,
                                    const ustring& text,
                                    std::vector<ustring>& words);

void KernelWordPieceTokenizer_Tokenizer(const std::unordered_map<ustring, int32_t>& vocab,
                                        const ustring& suffix_indicator,
                                        const ustring& unk_token,
                                        const std::vector<ustring>& texts,
                                        std::vector<ustring>& tokens,
                                        std::vector<int32_t>& indices,
                                        std::vector<int64_t>& rows,
                                        const int64_t* existing_rows = nullptr,
                                        int64_t n_existing_rows = 0,
                                        int64_t max_input_chars_per_word = 200);
