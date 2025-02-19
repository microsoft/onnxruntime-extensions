// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>
#include <string_view>
#include "file_sys.h"
#include "nlohmann/json.hpp"

#include "tokenizer_common.h"

namespace ort_extensions {

enum class TokenType {
  kUnknown, kUnigram, kBPE
};

constexpr std::pair<const char*, TokenType> kTokenizerDict[] = {
  {"PreTrainedTokenizer", TokenType::kBPE},
  {"CLIPTokenizer", TokenType::kBPE},
  {"WhisperTokenizer", TokenType::kBPE},
  {"GemmaTokenizer", TokenType::kBPE},
  {"LlamaTokenizer", TokenType::kBPE},
  {"Phi3Tokenizer", TokenType::kBPE},
  {"CodeLlamaTokenizer", TokenType::kBPE},
  {"CodeGenTokenizer", TokenType::kBPE},
  {"GPT2Tokenizer", TokenType::kBPE},
  {"Qwen2Tokenizer", TokenType::kBPE},
  {"BaichuanTokenizer", TokenType::kBPE},
  {"GPTNeoXTokenizer", TokenType::kBPE},

  {"", TokenType::kUnigram},
  {"T5Tokenizer", TokenType::kUnigram},
  {"ChatGLMTokenizer", TokenType::kUnigram},
  {"XLMRobertaTokenizer", TokenType::kUnigram}
};


// TokenJsonConfig: Handles loading and parsing of JSON configuration files for tokenizers
class TokenJsonConfig final {
 public:
  static constexpr const char* kDefaultVocabFile = "tokenizer.json";

  TokenJsonConfig() {}
  ~TokenJsonConfig() {}
  using json = nlohmann::json;
  using json_pointer = nlohmann::json_pointer<std::string>;
  json added_tokens_decoder;

 public:
  OrtxStatus AppendModuleJson(json& json_config) {
    auto iter = module_json_.find(tokenizer_class_);
    if (iter != module_json_.end()) {
      json json_module = json::parse(iter->second, nullptr, false, true);
      if (json_module.is_discarded()) {
        return OrtxStatus(kOrtxErrorInternal, "Failed to parse tokenizer module json.");
      }
      json_config.update(json_module);
    }

    return {};
  }

  OrtxStatus ParseTokensFromConfig(const json& json_config) {
    if (tokenizer_class_.empty()) {
      // Only the legacy fairseq/xlm-roberta tokenizer config doesn't have tokenizer_class.
      // Add ugly hack to handle it.
      add_bos_token_ = true;
      add_eos_token_ = true;
      bos_token_ = "<s>";
      eos_token_ = "</s>";
      unk_token_ = "<unk>";
      return {};
    }

    clean_up_tokenization_spaces_ = json_config.value("clean_up_tokenization_spaces", false);

    auto parse_token = [](const json& config, const std::string& key, std::string& token) {
      auto iter = config.find(key);
      if (iter != config.end() && !iter->is_null()) {
        if (iter->is_object()) {
          token = iter->value("content", "");
        } else {
          token = config.value(key, "");
        }
      }
    };

    parse_token(json_config, "bos_token", bos_token_);
    parse_token(json_config, "eos_token", eos_token_);
    parse_token(json_config, "unk_token", unk_token_);

    auto pad_iter = json_config.find("pad_token");
    if (pad_iter != json_config.end() && pad_iter->is_string()) {
      pad_token_ = json_config.value("pad_token", "");
    }

    add_bos_token_ = json_config.value("add_bos_token", false);
    add_eos_token_ = json_config.value("add_eos_token", false);
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
    } else {
      auto ifs = std::make_unique<std::ifstream>(vocab_path_);
      if (!ifs->is_open()) {
        return OrtxStatus(extError_t::kOrtxErrorInvalidArgument, vocab_path_ + ": does not exist.");
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
    auto status = AppendModuleJson(json_config);
    if (!status.IsOk()) {
      return status;
    }
    return ParseTokensFromConfig(json_config);
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

    // Store added_tokens_decoder to add any missed tokens into added_tokens in UpdateTokenizer 
    added_tokens_decoder = json_config.value("added_tokens_decoder", json::object());

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
    auto status = AppendModuleJson(json_config);
    if (!status.IsOk()) {
      return status;
    }
    return ParseTokensFromConfig(json_config);
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

  AddedTokenMap added_tokens_;

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

  static TokenType GetTokenType(const std::string& tok) {
    static const std::unordered_map<std::string_view, TokenType> dict {
        std::begin(kTokenizerDict), std::end(kTokenizerDict) };

    std::string_view tok_class(tok);
    auto pos = tok_class.find("Fast");
    if (pos != std::string_view::npos && pos + 4 == tok_class.size()) {
      tok_class.remove_suffix(4);
    }

    auto iter = dict.find(tok_class);
    return iter == dict.end() ? TokenType::kUnknown : iter->second;
  }

 private:
  void LoadAddedTokens(const json& tok_json) {
    auto added_tokens = tok_json.find("added_tokens");
    if (added_tokens != tok_json.end()) {
      for (const auto& token : *added_tokens) {
        auto tok_extended = ParseAddedToken(token);
        // insert the token into the unordered_map
        added_tokens_.emplace(ustring(tok_extended.content_), tok_extended);
      }
    }
  }

  std::string vocab_path_;
  const OrtxTokenizerBlob* blob_{nullptr};
  const std::map<std::string, std::string> module_json_ = {
      {"ChatGLMTokenizer", "{\"add_bos_token\"  : false, \"add_eos_token\": false}"},
      {"T5Tokenizer", "{\"add_bos_token\"  : false, \"add_eos_token\": true}"}};
};

}  // namespace ort_extensions
