// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "ustring.h"
#include "string_utils.h"
#include "string_tensor.h"

class BertTokenizerDecoder {
 public:
  BertTokenizerDecoder(std::string vocab, std::string unk_token, std::string sep_token, std::string pad_token,
                       std::string cls_token, std::string mask_token, std::string suffix_indicator);
  std::string Decode(const std::vector<int64_t>& ids, bool skip_special_tokens, bool clean_up_tokenization_spaces);

 private:
  std::string unk_token_;
  int32_t unk_token_id_ = -1;
  int32_t sep_token_id_ = -1;
  int32_t pad_token_id_ = -1;
  int32_t cls_token_id_ = -1;
  int32_t mask_token_id_ = -1;
  std::string suffix_indicator_;
  std::vector<std::string_view> vocab_;
  std::string raw_vocab_;
  std::vector<bool> is_substr_;

  bool RemoveTokenizeSpace(int64_t pre_token_id, int64_t new_token_id);
};

struct KernelBertTokenizerDecoder {
  
  template<typename T>
  KernelBertTokenizerDecoder(const T& dict) {
    //std::string vocab = ort_.KernelInfoGetAttribute<std::string>(&info, "vocab_file");
    std::string vocab = dict.TryToGetAttributeWithDefault("vocab_file", std::string(""));
    std::string unk_token = dict.TryToGetAttributeWithDefault("unk_token", std::string("[UNK]"));
    std::string sep_token = dict.TryToGetAttributeWithDefault("sep_token", std::string("[SEP]"));
    std::string pad_token = dict.TryToGetAttributeWithDefault("pad_token", std::string("[PAD]"));
    std::string cls_token = dict.TryToGetAttributeWithDefault("cls_token", std::string("[CLS]"));
    std::string mask_token = dict.TryToGetAttributeWithDefault("mask_token", std::string("[MASK]"));
    std::string suffix_indicator = dict.TryToGetAttributeWithDefault("suffix_indicator", std::string("##"));

    use_indices_ = dict.TryToGetAttributeWithDefault("use_indices", false);
    skip_special_tokens_ = dict.TryToGetAttributeWithDefault("skip_special_tokens", false);
    clean_up_tokenization_spaces_ = dict.TryToGetAttributeWithDefault("clean_up_tokenization_spaces", true);

    decoder_ = std::make_shared<BertTokenizerDecoder>(vocab, unk_token, sep_token, pad_token,
                                                      cls_token, mask_token, suffix_indicator);
  }

  void Compute(const ortc::Tensor<int64_t>& ids,
               const ortc::Tensor<int64_t>& positions,
               ortc::Tensor<std::string>& output) const;

 private:
  std::shared_ptr<BertTokenizerDecoder> decoder_;
  bool use_indices_;
  bool skip_special_tokens_;
  bool clean_up_tokenization_spaces_;
};
