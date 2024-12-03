// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "file_sys.h"
#include "nlohmann/json.hpp"

#include "tokenizer_common.h"

namespace ort_extensions {

// TokenJsonConfig: Handles loading and parsing of JSON configuration files for tokenizers
class TokenJsonConfig final {
 public:
  static constexpr const char* kDefaultVocabFile = "tokenizer.json";

  TokenJsonConfig() {}
  ~TokenJsonConfig() {}
  using json = nlohmann::json;
  using json_pointer = nlohmann::json_pointer<std::string>;

 public:
  OrtxStatus ParseTokensFromConfig(const json& json_config) {
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

  OrtxStatus OpenVocabFile(std::unique_ptr<std::istream>& vocab_stream) const {
    if (blob_ != nullptr) {
      if (blob_->vocab_blob_len == 0) {
        if (blob_->raw_model_blob_len == 0) {
          return OrtxStatus(kOrtxErrorInvalidArgument, "vocab_blob_len and raw_model_blob_len are both 0.");
        }
        std::string vocab_str(blob_->raw_model_blob, blob_->raw_model_blob_len);
        vocab_stream = std::make_unique<std::istringstream>(vocab_str);
      } else {
        if (blob_->raw_model_blob_len > 0) {
          return OrtxStatus(kOrtxErrorInvalidArgument, "vocab_blob_len and raw_model_blob_len are both non-zero.");
        }
        std::string vocab_str(blob_->vocab_json_blob, blob_->vocab_blob_len);
        vocab_stream = std::make_unique<std::istringstream>(vocab_str);
      }
    }
    else {
      auto ifs = std::make_unique<std::ifstream>(vocab_path_);
      if (!ifs->is_open()) {
        return OrtxStatus(extError_t::kOrtxErrorInvalidArgument, vocab_path_ +  ": does not exist.");
      }
      vocab_stream = std::move(ifs);
    }

    return {}; 
  }

  OrtxStatus LoadFromBlob(const OrtxTokenizerBlob& blob) {
    std::string config_str(blob.config_json_blob, blob.config_blob_len);
    std::istringstream config_ifs(config_str);
    json json_config = json::parse(config_ifs, nullptr, false, true);
    if (json_config.is_discarded()) {
      return OrtxStatus(kOrtxErrorInvalidArgument, "Failed to parse config json.");
    }

    if (blob.token_module_blob_len > 0) {
      std::string tokenizer_str(blob.token_module_blob, blob.token_module_blob_len);
      std::istringstream tokenizer_ifs(tokenizer_str);
      json json_tokenizer = json::parse(tokenizer_ifs, nullptr, false, true);
      if (json_tokenizer.is_discarded()) {
        return OrtxStatus(kOrtxErrorInvalidArgument, "Failed to parse tokenizer json.");
      }
      LoadAddedTokens(json_tokenizer);
      json_config.update(json_tokenizer);
    }

    blob_ = &blob;
    model_max_length_ = json_config.value("model_max_length", 1e+30);
    std::string tiktoken_file = json_config.value("tiktoken_file", "");
    if (!tiktoken_file.empty()) {
      if (blob.raw_model_blob_len == 0) {
        return OrtxStatus(kOrtxErrorInvalidArgument, "missing tiktoken file content in blob.raw_model_blob.");
      }
    }

    tokenizer_class_ = json_config.value("tokenizer_class", "");
    if (!tokenizer_class_.empty()) {
      return ParseTokensFromConfig(json_config);
    }

    return {};
  }

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

    json json_config = json::parse(ifs, nullptr, false, true);
    if (json_config.is_discarded()) {
      return OrtxStatus(kOrtxErrorInvalidArgument, "Failed to parse config json.");
    }

    auto module_cfg = tok_dir / "tokenizer_module.json";
    if (module_cfg.exists()) {
      std::ifstream module_ifs = module_cfg.open();
      json json_module = json::parse(module_ifs, nullptr, false, true);
      if (json_module.is_discarded()) {
        return OrtxStatus(kOrtxErrorInvalidArgument, "Failed to parse tokenizer module json.");
      }
      LoadAddedTokens(json_module);
      json_config.update(json_module);
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
    if (!tokenizer_class_.empty()) {
      return ParseTokensFromConfig(json_config);
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

  std::vector<ort_extensions::AddedToken> added_tokens_;

  static AddedToken ParseAddedToken(const json& token) {
    AddedToken added_token;
    added_token.id_ = token.value("id", 0);
    added_token.token_type_ = token.value("__type", "");
    added_token.content_ = token.value("content", "");
    added_token.lstrip_ = token.value("lstrip", false);
    added_token.normalized_ = token.value("normalized", false);
    added_token.rstrip_ = token.value("rstrip", false);
    added_token.single_word_ = token.value("single_word", false);
    added_token.special_ = token.value("special", false);
    return added_token;
  }


 private:
  void LoadAddedTokens(const json& tok_json) {
    auto added_tokens = tok_json.find("added_tokens");
    if (added_tokens != tok_json.end()) {
      for (const auto& token : *added_tokens) {
        added_tokens_.emplace_back(ParseAddedToken(token));
      }
    }
  }

  std::string vocab_path_;
  const OrtxTokenizerBlob* blob_{nullptr};
};

}  // namespace ort_extensions
