// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "file_sys.h"
#include "nlohmann/json.hpp"

#include "bpe_types.h"

namespace ort_extensions::bpe {

class TokenJsonConfig final {
 public:
  TokenJsonConfig() {}
  ~TokenJsonConfig() {}
  using json = nlohmann::json;
  using json_pointer = nlohmann::json_pointer<std::string>;

 public:
  OrtxStatus Load(const std::string& json_path) {
    if (json_path.empty()) {
      return OrtxStatus(kOrtxErrorInvalidArgument, "json_path is empty.");
    }

    auto file_path = path(json_path) / "tokenizer_config.json";
    std::ifstream ifs = file_path.open();
    if (!ifs.is_open()) {
      return OrtxStatus(kOrtxErrorInvalidFile, "Failed to open a json file: " + file_path.string());
    }

    vocab_path_ = (path(json_path) / "tokenizer.json").string();
    nlohmann::json json_config = nlohmann::json::parse(ifs);
    add_bos_token_ = json_config.value("add_bos_token", false);
    add_eos_token_ = json_config.value("add_eos_token", false);
    clean_up_tokenization_spaces_ = json_config.value("clean_up_tokenization_spaces", false);
    model_max_length_ = json_config.value("model_max_length", 1e+30);

    tokenizer_class_ = json_config.value("tokenizer_class", "");

    auto tok_iter = json_config.find("bos_token");
    if (tok_iter != json_config.end() && !tok_iter->is_null()) {
      if (tok_iter->is_object()) {
        bos_token_ = tok_iter->value("content", "");
        eos_token_ = json_config.value("/eos_token/content"_json_pointer, "");
        unk_token_ = json_config.value("/unk_token/content"_json_pointer, "");
      } else {
        bos_token_ = json_config.value("bos_token", "");
        eos_token_ = json_config.value("eos_token", "");
        unk_token_ = json_config.value("unk_token", "");
      }
    }

    auto pad_iter = json_config.find("pad_token");
    if (pad_iter != json_config.end() && pad_iter->is_string()) {
      pad_token_ = json_config.value("pad_token", "");
    }

    if (tokenizer_class_.empty()) {
      return OrtxStatus(kOrtxErrorCorruptData, "Failed to get tokenizer class.");
    }

    return {};
  }

  const std::string& GetVocabDataFile() const { return vocab_path_; }

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

 private:
  std::string vocab_path_;
};

}  // namespace ort_extensions::bpe
