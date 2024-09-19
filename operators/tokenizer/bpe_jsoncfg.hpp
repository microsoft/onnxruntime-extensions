// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "file_sys.h"
#include "nlohmann/json.hpp"

#include "tokjson_types.h"

namespace ort_extensions::bpe {

class TokenJsonConfig final {
 public:
  static constexpr const char* kDefaultVocabFile = "tokenizer.json";

  TokenJsonConfig() {}
  ~TokenJsonConfig() {}
  using json = nlohmann::json;
  using json_pointer = nlohmann::json_pointer<std::string>;

 public:
  OrtxStatus Load(const std::string& json_path) {
    if (json_path.empty()) {
      return OrtxStatus(kOrtxErrorInvalidArgument, "json_path is empty.");
    }

    ortx::path tok_dir(json_path);
    ortx::path vocab_path(json_path);
    ortx::path tok_path_obj(json_path);
    if (tok_path_obj.is_directory()) {
      vocab_path = tok_dir / kDefaultVocabFile;
    } else {
      if (!tok_path_obj.exists()) {
        return OrtxStatus(kOrtxErrorInvalidFile, "Invalid file: " + tok_path_obj.string());
      }

      tok_dir = ortx::path(tok_path_obj.parent_path());
    }

    auto config_path = tok_dir / "tokenizer_config.json";
    std::ifstream ifs = config_path.open();
    if (!ifs.is_open()) {
      return OrtxStatus(kOrtxErrorInvalidFile, "Failed to open a json file: " + config_path.string());
    }

    nlohmann::json json_config = nlohmann::json::parse(ifs);
    auto module_cfg = tok_dir / "tokenizer_module.json";
    if (module_cfg.exists()) {
      module_path_ = module_cfg.string();
      std::ifstream module_ifs = module_cfg.open();
      nlohmann::json module_config = nlohmann::json::parse(module_ifs);
      json_config.update(module_config);
    }

    model_max_length_ = json_config.value("model_max_length", 1e+30);
    std::string tiktoken_file = json_config.value("tiktoken_file", "");
    if (!tiktoken_file.empty()) {
      auto tktok_path = tok_dir / tiktoken_file;
      if (tktok_path.exists()) {
        vocab_path_ = tktok_path.string();
      } else {
        return OrtxStatus(kOrtxErrorInvalidFile, "Invalid file: " + tiktoken_file);
      }
    } else {
      if (ortx::path(vocab_path).exists()) {
        vocab_path_ = vocab_path.string();
      } else {
        return OrtxStatus(kOrtxErrorInvalidFile, "Invalid file: " + vocab_path.string());
      }
    }

    tokenizer_class_ = json_config.value("tokenizer_class", "");
    if (tokenizer_class_.empty()) {
      return {};
    }

    add_bos_token_ = json_config.value("add_bos_token", false);
    add_eos_token_ = json_config.value("add_eos_token", false);
    clean_up_tokenization_spaces_ = json_config.value("clean_up_tokenization_spaces", false);

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

    return {};
  }

  const std::string& GetVocabDataFile() const { return vocab_path_; }

  const std::string& GetTikTokenModuleFile() const { return module_path_; }

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
  std::string module_path_;
};

}  // namespace ort_extensions::bpe
