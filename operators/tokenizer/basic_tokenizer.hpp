// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"
#include "ustring.h"

class BasicTokenizer {
 public:
  BasicTokenizer(bool do_lower_case, bool tokenize_chinese_chars, bool strip_accents, bool tokenize_punctuation,
                 bool remove_control_chars);
  std::vector<ustring> Tokenize(ustring text);

 private:
  bool do_lower_case_;
  bool strip_accents_;
  bool tokenize_chinese_chars_;
  bool tokenize_punctuation_;
  bool remove_control_chars_;
};

struct KernelBasicTokenizer{
  template <typename T>
  KernelBasicTokenizer(const T& dict) {
    bool do_lower_case = dict.TryToGetAttributeWithDefault("do_lower_case", true);
    bool tokenize_chinese_chars = dict.TryToGetAttributeWithDefault("tokenize_chinese_chars", true);
    bool strip_accents = dict.TryToGetAttributeWithDefault("strip_accents", false);
    bool tokenize_punctuation = dict.TryToGetAttributeWithDefault("tokenize_punctuation", false);
    bool remove_control_chars = dict.TryToGetAttributeWithDefault("remove_control_chars", true);

    tokenizer_ = std::make_shared<BasicTokenizer>(do_lower_case, tokenize_chinese_chars, strip_accents,
                                                  tokenize_punctuation, remove_control_chars);
  }

  void Compute(std::string_view input,
               ortc::Tensor<std::string>& output) const;

 private:
  std::shared_ptr<BasicTokenizer> tokenizer_;
};
