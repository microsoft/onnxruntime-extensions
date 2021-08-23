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

class BertTokenizer {
 public:
  BertTokenizer(std::string vocab, bool do_lower_case, bool do_basic_tokenize,
                     ustring unk_token, ustring sep_token, ustring pad_token, ustring  cls_token,
                     ustring mask_token, bool tokenize_chinese_chars, bool strip_accents,
                     ustring suffix_indicator, int64_t max_input_chars_per_word);
  std::vector<int64_t> Tokenize(std::string);
 private:
  std::unordered_map<ustring, int32_t> vocab_;
  bool do_lower_case_;
  std::shared_ptr<BasicTokenizer> basic_tokenizer_;
  ustring unk_token_;
  ustring sep_token_;
  ustring pad_token_;
  ustring cls_token_;
  ustring mask_token_;
  ustring suffix_indicator_;
  bool do_basic_tokenize_;
  bool strip_accents_;
};