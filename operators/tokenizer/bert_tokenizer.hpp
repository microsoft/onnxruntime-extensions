// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "ustring.h"
#include "string_utils.h"
#include "string_tensor.h"
#include "basic_tokenizer.hpp"

#include <unordered_map>
#include <list>

class BertTokenizerVocab final {
 public:
  explicit BertTokenizerVocab(std::string_view vocab);
  bool FindToken(const ustring& token);
  bool FindTokenId(const ustring& token, int32_t& token_id);
  int32_t FindTokenId(const ustring& token);

 private:
  std::string raw_vocab_;
  std::unordered_map<std::string_view, int32_t> vocab_;
};

class TruncateStrategy final {
 public:
  explicit TruncateStrategy(std::string_view strategy_name);
  void Truncate(std::vector<int64_t>& ids, int32_t max_len);
  void Truncate(std::vector<int64_t>& ids1, std::vector<int64_t>& ids2, int32_t max_len);

 private:
  enum TruncateStrategyType {
    LONGEST_FIRST,
    ONLY_FIRST,
    ONLY_SECOND,
    LONGEST_FROM_BACK
  } strategy_;
};

// TODO: merge with the implementation of word piece tokenizer
class WordpieceTokenizer final {
 public:
  WordpieceTokenizer(
      std::shared_ptr<BertTokenizerVocab> vocab, ustring unk_token,
      ustring suffix_indicator, int max_input_chars_per_word = 100);
  using OffsetMappingType = std::list<std::pair<size_t, size_t>>;
  std::vector<ustring> Tokenize(const ustring& text, std::list<OffsetMappingType>& offset_map,
                                bool compute_offset_mapping);
  std::vector<ustring> Tokenize(const std::vector<ustring>& tokens, std::list<OffsetMappingType>& offset_map,
                                bool compute_offset_mapping);
  std::vector<int64_t> Encode(const std::vector<ustring>& tokens);

 private:
  int64_t max_input_chars_per_word_;
  ustring suffix_indicator_;
  ustring unk_token_;
  int32_t unk_token_id_;
  std::shared_ptr<BertTokenizerVocab> vocab_;

  void GreedySearch(const ustring& token, std::vector<ustring>& tokenized_result);
};

class BertTokenizer final {
 public:
  BertTokenizer(const std::string& vocab, bool do_lower_case, bool do_basic_tokenize,
                ustring unk_token, ustring sep_token, ustring pad_token, ustring cls_token,
                ustring mask_token, bool tokenize_chinese_chars, bool strip_accents,
                ustring suffix_indicator, int32_t max_len, const std::string& truncation_strategy);
  using OffsetMappingType = std::list<std::pair<size_t, size_t>>;
  std::vector<ustring> Tokenize(const ustring& text, std::list<OffsetMappingType>& offset_map,
                                bool compute_offset_mapping);
  std::vector<int64_t> Encode(const std::vector<ustring>& tokens);

  void Truncate(std::vector<int64_t>& ids);
  void Truncate(std::vector<int64_t>& ids1, std::vector<int64_t>& ids2);

  std::vector<int64_t> AddSpecialToken(const std::vector<int64_t>& ids);
  std::vector<int64_t> AddSpecialToken(const std::vector<int64_t>& ids1, const std::vector<int64_t>& ids2);
  std::vector<int64_t> GenerateTypeId(const std::vector<int64_t>& ids);
  std::vector<int64_t> GenerateTypeId(const std::vector<int64_t>& ids1, const std::vector<int64_t>& ids2);

 private:
  int32_t unk_token_id_ = 0;
  int32_t sep_token_id_ = 0;
  int32_t pad_token_id_ = 0;
  int32_t cls_token_id_ = 0;
  int32_t mask_token_id_ = 0;
  int32_t max_length_ = 0;
  bool do_basic_tokenize_ = false;
  std::unique_ptr<TruncateStrategy> truncate_;
  std::shared_ptr<BertTokenizerVocab> vocab_;
  std::unique_ptr<BasicTokenizer> basic_tokenizer_;
  std::unique_ptr<WordpieceTokenizer> wordpiece_tokenizer_;
};

struct KernelBertTokenizer : BaseKernel {
  KernelBertTokenizer(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Tensor<std::string>& input,
               ortc::Tensor<int64_t>& output,   // input_ids
               ortc::Tensor<int64_t>& output1,  // token_type_ids
               ortc::Tensor<int64_t>& output2,  // attention_mask
               std::optional<ortc::Tensor<int64_t>*> offset_mapping) const;
  using OffsetMappingType = std::list<std::pair<size_t, size_t>>;

 protected:
  std::unique_ptr<BertTokenizer> tokenizer_;
};

struct KernelHfBertTokenizer : KernelBertTokenizer {
  KernelHfBertTokenizer(const OrtApi& api, const OrtKernelInfo& info);
  using OffsetMappingType = std::list<std::pair<size_t, size_t>>;
  void Compute(const ortc::Tensor<std::string>& input,
               ortc::Tensor<int64_t>& output,                  // input_ids
               ortc::Tensor<int64_t>& output1,                 // attention_mask
               std::optional<ortc::Tensor<int64_t>*> output2,  // token_type_ids
               std::optional<ortc::Tensor<int64_t>*> offset_mapping) const;
};
