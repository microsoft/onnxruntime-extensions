// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <fstream>
#include <filesystem>
#include "ocos.h"
#include "status.h"
#include "nlohmann/json.hpp"

#include "bpe_types.h"

namespace ort_extensions::bpe {

class TokenJsonConfig final {
 public:
  TokenJsonConfig() {}
  ~TokenJsonConfig() {}

 public:
  OrtxStatus Load(const std::string& json_path) {
    if (json_path.empty()) {
      return OrtxStatus(kOrtxErrorInvalidArgument, "json_path is empty.");
    }

    auto file_path = std::filesystem::path(json_path) / "tokenizer_config.json";
    std::ifstream ifs(file_path);
    if (!ifs.is_open()) {
      return OrtxStatus(kOrtxErrorInvalidFile, "Failed to open a json file: " + file_path.string());
    }

    vocab_path_ = (std::filesystem::path(json_path) / "tokenizer.json").string();
    nlohmann::json json_config = nlohmann::json::parse(ifs);
    add_bos_token_ = json_config.value("add_bos_token", false);
    add_eos_token_ = json_config.value("add_eos_token", false);
    clean_up_tokenization_spaces_ = json_config.value("clean_up_tokenization_spaces", false);
    model_max_length_ = json_config.value("model_max_length", 1e+30);

    tokenizer_class_ = json_config.value("tokenizer_class", "");
    bos_token_ = json_config.value("bos_token", "");
    eos_token_ = json_config.value("eos_token", "");
    unk_token_ = json_config.value("unk_token", "");
    pad_token_ = json_config.value("pad_token", "");

    if (tokenizer_class_.empty()) {
      return OrtxStatus(kOrtxErrorCorruptData, "Failed to get tokenizer class.");
    }

    // auto added_tokens = json_config.find("added_tokens_decoder");
    // if (added_tokens != json_config.end()) {
    //   for (const auto& token : *added_tokens) {
    //     AddedToken added_token;
    //     added_token.token_type_ = token.value("type", "");
    //     added_token.content_ = token.value("content", "");
    //     added_token.lstrip_ = token.value("lstrip", false);
    //     added_token.normalized_ = token.value("normalized", false);
    //     added_token.rstrip_ = token.value("rstrip", false);
    //     added_token.single_word_ = token.value("single_word", false);

    //     if (added_token.token_type_ == "special_tokens") {
    //       if (added_token.content_ == unk_token_) {
    //         unk_token_item_ = added_token;
    //       } else if (added_token.content_ == bos_token_) {
    //         bos_token_item_ = added_token;
    //       } else if (added_token.content_ == eos_token_) {
    //         eos_token_item_ = added_token;
    //       }
    //     }
    //   }
    // }

    return {};
  }

  const std::string& GetVocabDataFile() const {
    return vocab_path_;
  }

 public:
  bool add_bos_token_{};
  bool add_eos_token_{};
  bool clean_up_tokenization_spaces_{};
  double model_max_length_{};

  std::string tokenizer_class_;
  std::string bos_token_;
  std::string eos_token_;
  std::string unk_token_;
  std::string pad_token_;

  // AddedToken bos_token_item_;
  // AddedToken eos_token_item_;
  // AddedToken unk_token_item_;

 private:
  std::string vocab_path_;
};

}  // namespace ort_extensions::bpe
