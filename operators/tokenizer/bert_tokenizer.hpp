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
  WordpieceTokenizer(const std::string& vocab, ustring unk_token, ustring suffix_indicator, int max_input_chars_per_word = 100);
  std::vector<ustring> Tokenize(const ustring& text);
  std::vector<ustring> Tokenize(const std::vector<ustring>& tokens);
  std::vector<int64_t> Encode(const std::vector<ustring>& tokens);
 private:
  int64_t max_input_chars_per_word_;
  ustring suffix_indicator_;
  ustring unk_token_;
  std::unordered_map<ustring, int32_t> vocab_;

  void GreedySearch(const ustring& token, std::vector<ustring> tokenized_result);
};

class BertTokenizer {
 public:
  BertTokenizer(std::string vocab, bool do_lower_case, bool do_basic_tokenize,
                     ustring unk_token, ustring sep_token, ustring pad_token, ustring  cls_token,
                     ustring mask_token, bool tokenize_chinese_chars, bool strip_accents,
                     ustring suffix_indicator, int64_t max_input_chars_per_word);
  std::vector<ustring> Tokenize(const ustring& text);
  std::vector<int64_t> Encode(const std::vector<ustring>& tokens);
 private:
  std::unordered_map<ustring, int32_t> vocab_;

  ustring unk_token_;
  ustring sep_token_;
  ustring pad_token_;
  ustring cls_token_;
  ustring mask_token_;
  bool do_basic_tokenize_;
  std::shared_ptr<BasicTokenizer> basic_tokenizer_;
  std::shared_ptr<WordpieceTokenizer> wordpiece_tokenizer_;
};