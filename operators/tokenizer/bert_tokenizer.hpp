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

// TODO: merge with the implementation of word piece tokenizer
class WordpieceTokenizer{
 public:
  WordpieceTokenizer(std::shared_ptr<std::unordered_map<ustring, int32_t>> vocab, ustring unk_token, ustring suffix_indicator, int max_input_chars_per_word = 100);
  std::vector<ustring> Tokenize(const ustring& text);
  std::vector<ustring> Tokenize(const std::vector<ustring>& tokens);
  std::vector<int64_t> Encode(const std::vector<ustring>& tokens);
 private:
  int64_t max_input_chars_per_word_;
  ustring suffix_indicator_;
  ustring unk_token_;
  int64_t unk_token_id_;
  std::shared_ptr<std::unordered_map<ustring, int32_t>> vocab_;

  void GreedySearch(const ustring& token, std::vector<ustring>& tokenized_result);
};

class BertTokenizer {
 public:
  BertTokenizer(std::string vocab, bool do_lower_case, bool do_basic_tokenize,
                     ustring unk_token, ustring sep_token, ustring pad_token, ustring  cls_token,
                     ustring mask_token, bool tokenize_chinese_chars, bool strip_accents,
                     ustring suffix_indicator);
  std::vector<ustring> Tokenize(const ustring& text);
  std::vector<int64_t> Encode(const std::vector<ustring>& tokens);
  std::vector<int64_t> AddSpecialToken(const std::vector<int64_t>& ids);
  std::vector<int64_t> AddSpecialToken(const std::vector<int64_t>& ids1, const std::vector<int64_t>& ids2);
  std::vector<int64_t> GenerateTypeId(const std::vector<int64_t>& ids);
  std::vector<int64_t> GenerateTypeId(const std::vector<int64_t>& ids1, const std::vector<int64_t>& ids2);
 private:
  int32_t unk_token_id_;
  int32_t sep_token_id_;
  int32_t pad_token_id_;
  int32_t cls_token_id_;
  int32_t mask_token_id_;
  bool do_basic_tokenize_;
  std::shared_ptr<std::unordered_map<ustring, int32_t>> vocab_;
  std::shared_ptr<BasicTokenizer> basic_tokenizer_;
  std::shared_ptr<WordpieceTokenizer> wordpiece_tokenizer_;

  int32_t FindSpecialToken(ustring token);
};



class TruncateStrategy {
  TruncateStrategy(std::string strategy);
  void Truncate(std::vector<int64_t>& ids, int64_t max_len);
  void Truncate(std::vector<int64_t>& input1, std::vector<int64_t>& input2, int64_t max_len);

 private:
  enum TruncateStrategyType{
    LONGEST_FIRST,
    ONLY_FIRST,
    ONLY_SECOND,
    LONGEST_FROM_BACK
  }strategy_;
};


struct KernelBertTokenizer : BaseKernel {
  KernelBertTokenizer(OrtApi api,  const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);
 private:
  std::shared_ptr<BertTokenizer> tokenizer_;
};

struct CustomOpBertTokenizer : Ort::CustomOpBase<CustomOpBertTokenizer, KernelBertTokenizer> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};