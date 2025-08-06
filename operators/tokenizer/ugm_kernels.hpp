// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// The implementation is inspired by llama.cpp ugm tokenizer and huggingface FastTokenizer

#pragma once

#include <map>
#include <set>
#include <list>
#include <string>
#include <string_view>
#include <vector>
#include <cfloat>
#include <functional>
#include <unordered_map>
#include <cwctype>
#include <locale>

#include "ortx_tokenizer.h"
#include "ext_status.h"
#include "op_def_struct.h"
#include "base64.h"
#include "ustring.h"
#include "narrow.h"
#include "nlohmann/json.hpp"
#include "trietree.hpp"
#include "tokenizer_jsconfig.hpp"
#include "case_encoder.h"

namespace ort_extensions {

class SpmUgmDecoder;  // forward declaration

struct SpmUgmTokenizer {
  using json = nlohmann::json;
  using VocabTrieTree = ort_extensions::TrieTree<char, extTokenId_t, -1>;
  using Vocab = std::unordered_map<std::string, std::tuple<extTokenId_t, double>>;

  SpmUgmTokenizer() = default;

  OrtxStatus LoadSpecialTokens(const json& token_json) {
    auto special_tokens = token_json.find("added_tokens");
    if (special_tokens != token_json.end()) {
      for (const auto& token : special_tokens->items()) {
        auto id = token.value()["id"].get<extTokenId_t>();
        bool is_special = token.value()["special"].get<bool>();
        if (is_special) {
          special_token_ids_.insert(id);
        }
        auto word = token.value()["content"].get<std::string>();
        user_defined_token_matcher_.Add(word, 0, id);
      }
    }

    return {};
  }

  OrtxStatus LoadCharsMap(const json& j_vocab) {
    auto normalizer = j_vocab.find("normalizer");
    std::string charsmap;
    if (normalizer != j_vocab.end()) {
      auto iter = normalizer->find("precompiled_charsmap");
      if (iter != normalizer->end()) {
        charsmap = iter->get<std::string>();
      } else {
        auto iter = normalizer->find("normalizers");  // v2 schema
        if (iter != normalizer->end()) {
          for (const auto& normalizer : iter->items()) {
            if (normalizer.value().contains("type")) {
              auto type = normalizer.value()["type"].get<std::string>();
              if (type == "Precompiled") {
                charsmap = normalizer.value()["precompiled_charsmap"].get<std::string>();
                break;
              }
            }
          }
        }
      }
    }

    if (!charsmap.empty()) {
      if (!base64_decode(charsmap, charsmap_data_)) {
        return OrtxStatus(extError_t::kOrtxErrorCorruptData, "Failed to decode charsmap.");
      }

      // std::cout << "charsmap size: " << charsmap_data_.size() << std::endl;
      // for (size_t i = 0; i < charsmap_data_.size() && i < 100; ++i) {
      //   std::cout << int(charsmap_data_[i]) << " ";
      // }

      size_t charsmap_offset = 0;

      // First four bytes of precompiled_charsmap contains length of binary
      // blob containing XOR-compressed compact double array (XCDA) entries
      uint32_t xcda_blob_size = *(const uint32_t*)&charsmap_data_[0];
      charsmap_offset += sizeof(xcda_blob_size);
      if (xcda_blob_size + charsmap_offset >= charsmap_data_.size()) {
        return OrtxStatus(extError_t::kOrtxErrorCorruptData, "Index out of array bounds in precompiled charsmap!");
      }

      // Next xcda_blob_size bytes contain entries of XOR-compressed compact
      // double array (XCDA). Each entry is bit-packed into a 32-bit integer.
      xcda_array_ = (const uint32_t*)&charsmap_data_[charsmap_offset];
      xcda_array_size_ = xcda_blob_size / sizeof(uint32_t);
      charsmap_offset += xcda_blob_size;

      // Remaining bytes of precompiled charsmap contain null-terminated
      // replacement strings for prefixes matched by the XCDA.
      prefix_replacements_ = reinterpret_cast<const char*>(&charsmap_data_[charsmap_offset]);
      prefix_replacements_size_ = charsmap_data_.size() - charsmap_offset;
    }

    return {};
  }

  OrtxStatus LoadConfig(const json& config) {
    auto pretokenizer_node = config.find("pretokenizer");
    if (pretokenizer_node != config.end()) {
      auto pretokenizers_node = pretokenizer_node->find("pretokenizers");
      if (pretokenizers_node != pretokenizer_node->end()) {
        for (const auto& pretokenizer : pretokenizers_node->items()) {
          if (pretokenizer.value().contains("type")) {
            auto type = pretokenizer.value()["type"].get<std::string>();
            if (type == "Metaspace") {
              tokenizer_escape_whitespaces_ = true;
            }
          }
          if (pretokenizer.value().contains("add_prefix_space")) {
            tokenizer_add_space_prefix_ = pretokenizer.value()["add_prefix_space"].get<bool>();
          }
        }
      }
    }
    return {};
  }

  OrtxStatus Load(const TokenJsonConfig& config) {
    add_bos_token_ = config.add_bos_token_;
    add_eos_token_ = config.add_eos_token_;
    bos_token_ = config.bos_token_;
    eos_token_ = config.eos_token_;
    unk_token_ = config.unk_token_;
    pad_token_ = config.pad_token_;

    if (config.tokenizer_class_ == "ChatGLMTokenizer") {
      chatglm_special_endings_ = true;
      tokenizer_remove_extra_whitespaces_ = false;
    }

    std::unique_ptr<std::istream> vocab_stream;
    auto status = config.OpenVocabFile(vocab_stream);
    if (!status.IsOk()) {
      return status;
    }

    nlohmann::json j_vocab = json::parse(*vocab_stream, nullptr, false, true);
    if (j_vocab.is_discarded()) {
      return OrtxStatus(extError_t::kOrtxErrorInvalidArgument, "Failed to parse vocabulary file.");
    }

    status = LoadConfig(j_vocab);
    if (!status.IsOk()) {
      return status;
    }

    status = LoadCharsMap(j_vocab);
    if (!status.IsOk()) {
      return status;
    }

    status = LoadSpecialTokens(j_vocab);
    if (!status.IsOk()) {
      return status;
    }

    auto model_node = j_vocab.find("model");
    if (model_node == j_vocab.end()) {
      return OrtxStatus(extError_t::kOrtxErrorInvalidArgument, "Model node not found in vocabulary file.");
    }

    auto unk_id_iter = model_node->find("unk_id");
    if (unk_id_iter != model_node->end()) {
      special_unk_id_ = unk_id_iter->get<extTokenId_t>();
    }

    auto vocab_node = model_node->find("vocab");
    if (vocab_node == model_node->end()) {
      return OrtxStatus(extError_t::kOrtxErrorInvalidArgument, "Vocabulary not found in model node.");
    }

    static std::string_view val_pattern = "<0xXY>";
    auto convert_hex_to_token = [](const std::string& str) {
      int val = -1;
      char char_3_4[3] = {0};
      if (str.size() != val_pattern.size()) {
        return val;
      }
      for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] != val_pattern[i]) {
          if (i == 3) {
            char_3_4[0] = str[i];
          } else if (i == 4) {
            char_3_4[1] = str[i];
          }
        }
      }
      if (char_3_4[0] < '0' || (char_3_4[0] > '9' && char_3_4[0] < 'A') || char_3_4[0] > 'F' || char_3_4[1] < '0' ||
          (char_3_4[1] > '9' && char_3_4[1] < 'A') || char_3_4[1] > 'F') {
        return val;
      }
      return std::stoi(char_3_4, nullptr, 16);
    };

    extTokenId_t id = 0;
    for (const auto& entry : vocab_node->items()) {
      auto score = entry.value()[1].get<double>();
      auto tkn = entry.value()[0].get<std::string>();
      int val = convert_hex_to_token(tkn);
      if (val != -1) {
        tkn = std::string(1, static_cast<unsigned char>(val));
      }
      if (chatglm_special_endings_) {
        if (tkn == "<n>") {
          tkn = "\n";
        } else if (tkn == "<|tab|>") {
          tkn = "\t";
        } else if (tkn.size() == 11) {  // length of "<|blank_x|>"
          auto blank_pos = tkn.find("<|blank_");
          if (blank_pos != std::string::npos) {
            auto num = tkn[blank_pos + 8] - '0';
            if (num >= 2 && num <= 9) {
              tkn.clear();
              tkn.reserve(num);
              for (; num != 0; num--) {
                tkn += "▁";
              }
            }
          }
        }
      }
      if (score != 0.0 || vocab_.count(tkn) == 0) {
        vocab_[tkn] = std::make_tuple(id++, score);
      }
    }

    scores_.resize(id);
    double min_score = DBL_MAX;
    for (const auto& entry : vocab_) {
      scores_[std::get<0>(entry.second)] = std::get<1>(entry.second);
      token_matcher_.Add(entry.first, 0, std::get<0>(entry.second));
      min_score = std::min<double>(min_score, std::get<1>(entry.second));
    }

    unknown_token_score_ = min_score - unknown_token_score_penalty_;

    if (config.tokenizer_class_ == "MarianTokenizer") {
      byte_fallback_ = true;
      tokenizer_add_space_prefix_ = true;
      tokenizer_remove_extra_whitespaces_ = false;
      tokenizer_treat_whitespace_as_suffix_ = true;
      add_eos_token_ = true;
      case_encoder_ = std::make_unique<normalizer::CaseEncoder>(tokenizer_remove_extra_whitespaces_);
      case_encoder_->SetNormalizer([this](std::string_view input) { return NmtNormalizePrefix(input); });
    }
    return status;
  }

  extTokenId_t GetTokenId(const std::string& token) const {
    auto iter = vocab_.find(token);
    if (iter == vocab_.end()) {
      return special_unk_id_;
    }
    return std::get<0>(iter->second);
  }

  OrtxStatus ComputeNoOp(const std::string& input, std::vector<extTokenId_t>& output,
                         bool add_special_tokens = true) const {
    std::string normalized;
    if (case_encoder_) {
      normalized = NmtNormalize(input);
    } else {
      normalized = Normalize(input);
    }
    size_t input_len = normalized.size();
    if (input_len == 0) {
      return {};
    }

    std::vector<struct BestTokenization> tokenization_results(input_len + 1, {0, -DBL_MAX, special_unk_id_});
    tokenization_results[0] = {0, 0, special_unk_id_};

    for (size_t input_offset = 0; input_offset < input_len;) {
      size_t prefix_offset = input_offset;
      size_t n_utf8_code_units = std::min<size_t>(ustring::UTF8Len(normalized[input_offset]), input_len - input_offset);

      bool single_codepoint_token_found = false;
      const struct BestTokenization& current_best = tokenization_results[input_offset];
      auto node = token_matcher_.Find(normalized[prefix_offset++]);

      while (prefix_offset <= input_len && node != NULL) {
        if (node->HasValue()) {
          if (prefix_offset - input_offset == n_utf8_code_units) {
            single_codepoint_token_found = true;
          }
          extTokenId_t token_id = node->Value();
          const auto& token_data = scores_[token_id];

          const double token_score = special_token_ids_.count(token_id) > 0 ? 0.0 : token_data;
          const double challenger_score = current_best.score_sum + token_score;
          struct BestTokenization& current_champ = tokenization_results[prefix_offset];
          if (challenger_score > current_champ.score_sum) {
            struct BestTokenization challenger = {input_offset, (float)challenger_score, token_id};
            current_champ = challenger;
          }
        }
        node = node->Find(normalized[prefix_offset++]);
      }

      if (!single_codepoint_token_found) {
        if (byte_fallback_) {
          prefix_offset -= 1;
          n_utf8_code_units = prefix_offset - input_offset;
        } else {
          const double challenger_score = current_best.score_sum + unknown_token_score_;
          prefix_offset = input_offset + n_utf8_code_units;
          struct BestTokenization& current_champ = tokenization_results[prefix_offset];
          if (challenger_score > current_champ.score_sum) {
            struct BestTokenization challenger = {input_offset, (float)challenger_score, special_unk_id_};
            current_champ = challenger;
          }
        }
      }

      input_offset += n_utf8_code_units;
    }

    output.reserve(input_len);
    bool is_prev_unknown = false;
    for (struct BestTokenization& tokenization = tokenization_results[input_len];;
         tokenization = tokenization_results[tokenization.input_offset]) {
      bool is_unknown = tokenization.token_id == special_unk_id_;
      if (!(is_prev_unknown && is_unknown)) {
        output.push_back(tokenization.token_id);
      }
      if (tokenization.input_offset == 0) {
        break;
      }
      is_prev_unknown = is_unknown;
    }

    // will be reversed
    if (add_bos_token_ && add_special_tokens) {
      output.push_back(GetTokenId(bos_token_));
    }
    std::reverse(output.begin(), output.end());
    if (chatglm_special_endings_) {
      auto unknown_token_id = GetTokenId(unk_token_);
      // remove the unknown token in the output ids
      output.erase(
          std::remove_if(output.begin(), output.end(), [this](extTokenId_t id) { return id == special_unk_id_; }),
          output.end());
      output.push_back(GetTokenId("[gMASK]"));
      output.push_back(GetTokenId("<sop>"));
    }

    if (add_eos_token_ && add_special_tokens) {
      output.push_back(GetTokenId(eos_token_));
    }

    return {};
  }

  OrtxStatus Compute(const ortc::Tensor<std::string>& input, ortc::Tensor<int64_t>& tokenize_output,
                     std::optional<ortc::Tensor<int64_t>*> attention_mask = std::nullopt,
                     std::optional<ortc::Tensor<int64_t>*> offset_mapping = std::nullopt,
                     std::optional<bool> add_special_tokens = true) const {
    if (attention_mask.has_value() || offset_mapping.has_value()) {
      return {kOrtxErrorInvalidArgument, "attention-mask or offset-mapping was supported in unigram tokenizer"};
    }

    // Update add_special_tokens
    bool append_special_tokens = true;
    if (add_special_tokens.has_value()) {
      append_special_tokens = add_special_tokens.value();
    }

    if (input.Shape().size() != 1) {
      return OrtxStatus(extError_t::kOrtxErrorInvalidArgument, "Input tensor must have rank 1.");
    }

    std::vector<extTokenId_t> ids_vec;
    auto status = ComputeNoOp(input.AsScalar(), ids_vec, append_special_tokens);
    if (status.IsOk()) {
      auto output_size = static_cast<int64_t>(ids_vec.size());
      int64_t* id_output = tokenize_output.Allocate({output_size});
      std::transform(ids_vec.begin(), ids_vec.end(), id_output,
                     [](extTokenId_t id) { return static_cast<int64_t>(id); });
    }

    return status;
  }

 private:
  struct NormalizationResult {
    const char* normalized;
    size_t normalized_len;
    size_t consumed_input;
  };

  std::string Normalize(const std::string& input) const {
    std::string normalized;
    normalized.reserve(input.size() * 3);

    const std::string space = tokenizer_escape_whitespaces_ ? std::string(spm_escaped_space) : " ";

    bool shall_prepend_space = !tokenizer_treat_whitespace_as_suffix_ && tokenizer_add_space_prefix_;
    bool shall_append_space = tokenizer_treat_whitespace_as_suffix_ && tokenizer_add_space_prefix_;
    bool shall_merge_spaces = tokenizer_remove_extra_whitespaces_;

    bool is_space_prepended = false;
    bool processing_non_ws = false;

    size_t input_len = input.size();

    for (size_t input_offset = 0; input_offset < input_len;) {
      auto norm_res = NormalizePrefix(input, input_offset);
      for (size_t i = 0; i < norm_res.normalized_len; i++) {
        char c = norm_res.normalized[i];
        if (c != ' ') {
          if (!processing_non_ws) {
            processing_non_ws = true;
            if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
              normalized.append(space);
              is_space_prepended = true;
            }
          }
          normalized.push_back(c);
        } else {
          if (processing_non_ws) {
            processing_non_ws = false;
          }
          if (!shall_merge_spaces) {
            normalized.append(space);
          }
        }
      }

      input_offset += norm_res.consumed_input;
    }

    if (shall_append_space) {
      normalized.append(space);
    }

    return normalized;
  }

  std::pair<std::string_view, int> NmtNormalizePrefix(std::string_view input_view) const {
    if (input_view.empty()) {
      return {"", 0};
    }

    size_t prefix_off = 0;
    auto user_defined_token_match = user_defined_token_matcher_.FindLongest(std::string(input_view), prefix_off);
    if (user_defined_token_match != user_defined_token_matcher_.kInvalidId_) {
      return {input_view.substr(0, prefix_off), static_cast<int>(prefix_off)};
    }

    size_t longest_prefix_length = 0;
    size_t longest_prefix_offset = 0;

    if (xcda_array_size_ > 0) {
      XcdaArrayView xcda_view(xcda_array_, xcda_array_size_);

      uint32_t node_index = 0;
      node_index = xcda_view.GetBase(node_index);
      for (size_t prefix_offset = 0; prefix_offset < input_view.size(); prefix_offset++) {
        unsigned char c = input_view[prefix_offset];
        if (c == 0) {
          break;
        }
        node_index ^= c;
        if (xcda_view.GetLcheck(node_index) != c) {
          break;
        }
        bool is_leaf = xcda_view.IsLeaf(node_index);
        node_index ^= xcda_view.GetBase(node_index);
        if (is_leaf) {
          longest_prefix_length = prefix_offset + 1;
          longest_prefix_offset = xcda_view.GetValue(node_index);
        }
      }
    }

    if (longest_prefix_length > 0) {
      if (longest_prefix_offset >= prefix_replacements_size_) {
        ORTX_CXX_API_THROW("[UgmTok]Index out of array bounds in precompiled charsmap!", ORT_RUNTIME_EXCEPTION);
      }
      const char* prefix_replacement = &prefix_replacements_[longest_prefix_offset];
      return {prefix_replacement, static_cast<int>(longest_prefix_length)};
    } else {
      // if yes, return this sequence unmodified
      size_t prefix_offset = ustring::UTF8Len(input_view[0]);
      if (prefix_offset <= input_view.size()) {
        return {input_view.substr(0, prefix_offset), static_cast<int>(prefix_offset)};
      }
    }

    return {"\xEF\xBF\xBD", 1};
  }

  std::string NmtNormalize(const std::string& input) const {
    std::string normalized;
    normalized.reserve(input.size() * 3);
    std::vector<size_t> norm_to_orig(input.size() * 3);

    const std::string space = tokenizer_escape_whitespaces_ ? std::string(spm_escaped_space) : " ";

    bool shall_prepend_space = !tokenizer_treat_whitespace_as_suffix_ && tokenizer_add_space_prefix_;
    bool shall_append_space = tokenizer_treat_whitespace_as_suffix_ && tokenizer_add_space_prefix_;
    bool shall_merge_spaces = tokenizer_remove_extra_whitespaces_;

    bool is_space_prepended = false;
    bool processing_non_ws = false;

    size_t input_len = input.size();

    std::string_view input_view(input);
    int consumed = 0;

    while (!input_view.empty()) {
      auto p = case_encoder_->NormalizePrefix(input_view);

      for (size_t i = 0; i < p.first.size(); i++) {
        char c = p.first[i];
        if (c != ' ') {
          if (!processing_non_ws) {
            processing_non_ws = true;
            if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
              normalized.append(space);
              is_space_prepended = true;
            }
          }
          normalized.push_back(c);
        } else {
          if (processing_non_ws) {
            processing_non_ws = false;
          }
          if (!shall_merge_spaces) {
            normalized.append(space);
          }
        }
      }

      consumed += p.second;
      input_view.remove_prefix(p.second);
    }

    case_encoder_->PostProcess(&normalized, &norm_to_orig);

    if (shall_append_space) {
      normalized.append(space);
    }

    return normalized;
  }

  /*
   * This structure is a view wrapper for XOR-compressed double array (XCDA)
   * See Shunsuke Kanda (2018). Space- and Time-Efficient String Dictionaries.
   * Each bit-packed entry contains:
   * - BASE array value in bits 10-30
   * - LCHECK array value in bits 0-7
   * - LEAF array value in bit 9
   * Entries containing indexes of replacement sequences have set bit 31
   */
  struct XcdaArrayView {
   public:
    XcdaArrayView(const uint32_t* xcda_array, size_t xcda_array_size)
        : xcda_array_(xcda_array), xcda_array_size_(xcda_array_size) {}
    uint32_t GetBase(size_t index) {
      uint32_t packed_node = GetNode(index);
      return (packed_node >> 10) << ((packed_node & (1U << 9)) >> 6);
    }
    uint32_t GetLcheck(size_t index) {
      uint32_t packed_node = GetNode(index);
      return packed_node & ((1U << 31) | 0xff);
    }
    bool IsLeaf(size_t index) {
      uint32_t packed_node = GetNode(index);
      return (packed_node >> 8) & 1;
    }
    uint32_t GetValue(size_t index) {
      uint32_t packed_node = GetNode(index);
      return packed_node & ((1U << 31) - 1);
    }

   private:
    uint32_t GetNode(size_t index) {
      if (index > xcda_array_size_) {
        ORTX_CXX_API_THROW("[UgmTok]Index out of array bounds in XCDA array!", ORT_RUNTIME_EXCEPTION);
      }
      return xcda_array_[index];
    }
    const uint32_t* xcda_array_;
    size_t xcda_array_size_;
  };

  NormalizationResult NormalizePrefix(const std::string& input, size_t input_offset) const {
    if (input_offset == input.size()) {
      return {&input[input_offset], 0, 0};
    }

    std::string prefix = input.substr(input_offset);
    size_t prefix_off = 0;
    auto user_defined_token_match = user_defined_token_matcher_.FindLongest(prefix, prefix_off);
    if (user_defined_token_match != user_defined_token_matcher_.kInvalidId_) {
      return {&input[input_offset], prefix_off + input_offset, prefix_off + input_offset};
    }

    size_t longest_prefix_length = 0;
    size_t longest_prefix_offset = 0;

    if (xcda_array_size_ > 0) {
      XcdaArrayView xcda_view(xcda_array_, xcda_array_size_);

      uint32_t node_index = 0;
      node_index = xcda_view.GetBase(node_index);
      for (size_t prefix_offset = input_offset; prefix_offset < input.size(); prefix_offset++) {
        unsigned char c = input[prefix_offset];
        if (c == 0) {
          break;
        }
        node_index ^= c;
        if (xcda_view.GetLcheck(node_index) != c) {
          break;
        }
        bool is_leaf = xcda_view.IsLeaf(node_index);
        node_index ^= xcda_view.GetBase(node_index);
        if (is_leaf) {
          longest_prefix_length = prefix_offset - input_offset + 1;
          longest_prefix_offset = xcda_view.GetValue(node_index);
        }
      }
    }

    if (longest_prefix_length > 0) {
      if (longest_prefix_offset >= prefix_replacements_size_) {
        ORTX_CXX_API_THROW("[UgmTok]Index out of array bounds in precompiled charsmap!", ORT_RUNTIME_EXCEPTION);
      }
      const char* prefix_replacement = &prefix_replacements_[longest_prefix_offset];
      return {prefix_replacement, strlen(prefix_replacement), longest_prefix_length};
    } else {
      // if yes, return this sequence unmodified
      size_t prefix_offset = input_offset + ustring::UTF8Len(input[input_offset]);
      if (prefix_offset <= input.size()) {
        return {&input[input_offset], prefix_offset - input_offset, prefix_offset - input_offset};
      }
    }

    return {"\xEF\xBF\xBD", 3, 1};
  }

  friend class SpmUgmDecoder;
  // escaped space symbol - U+2581 (Lower One Eighth Block)
  static constexpr double unknown_token_score_penalty_ = 10.0;

  std::vector<uint8_t> charsmap_data_;
  const char* prefix_replacements_ = NULL;
  size_t prefix_replacements_size_ = 0;

  const uint32_t* xcda_array_ = NULL;
  size_t xcda_array_size_ = 0;

  VocabTrieTree user_defined_token_matcher_;

  struct BestTokenization {
    size_t input_offset;
    double score_sum;
    extTokenId_t token_id;
  };

  extTokenId_t special_unk_id_ = -1;
  double unknown_token_score_{};

  Vocab vocab_;
  std::vector<double> scores_;
  std::set<extTokenId_t> special_token_ids_;
  VocabTrieTree token_matcher_;

 public:
  bool byte_fallback_ = false;
  bool chatglm_special_endings_ = false;
  bool tokenizer_escape_whitespaces_ = true;
  bool tokenizer_treat_whitespace_as_suffix_ = false;
  bool tokenizer_add_space_prefix_ = true;
  bool tokenizer_remove_extra_whitespaces_ = true;
  std::string bos_token_ = "<s>";
  std::string eos_token_ = "</s>";
  std::string pad_token_ = "<pad>";
  std::string unk_token_ = "<unk>";
  bool add_bos_token_{};  // add bos token
  bool add_eos_token_{};  // add eos token

  std::unique_ptr<normalizer::CaseEncoder> case_encoder_;
};

class SpmUgmDecoder {
 public:
  SpmUgmDecoder() {}

  OrtxStatus Load(const TokenJsonConfig& config, const SpmUgmTokenizer& tokenizer) {
    // fill the vocab_ with the default token
    vocab_.resize(tokenizer.scores_.size(), tokenizer.unk_token_);

    for (auto iter = tokenizer.vocab_.begin(); iter != tokenizer.vocab_.end(); ++iter) {
      vocab_[std::get<0>(iter->second)] = iter->first;
    }

    unknown_token_ = tokenizer.unk_token_;
    special_token_ids_ = tokenizer.special_token_ids_;
    tokenizer_add_space_prefix_ = tokenizer.tokenizer_add_space_prefix_;
    case_encoding_ = tokenizer.case_encoder_ != nullptr;
    return {};
  }

  OrtxStatus Compute(const ortc::Tensor<int64_t>& ids, ortc::Tensor<std::string>& output) const {
    const int64_t* p_ids = ids.Data();
    const auto& ids_dim = ids.Shape();
    std::vector<int64_t> output_dim = {1};
    if (ids_dim.size() > 1) {
      output_dim.resize(ids_dim.size() - 1);
      std::copy(ids_dim.begin(), ids_dim.begin() + ids_dim.size() - 1, output_dim.begin());
    }

    int64_t seq_len = ids_dim.back();
    size_t string_batch = ids.NumberOfElement() / seq_len;

    std::vector<std::string> decoded_strings;
    decoded_strings.reserve(string_batch);
    TokenizerDecodingState* state{};
    for (auto n = string_batch; n > 0; n--) {
      std::string text;
      for (int64_t i = 0; i < seq_len; ++i) {
        std::string token;
        Id2Token(ort_extensions::narrow<extTokenId_t>(p_ids[i]), token, &state);
        text += token;
      }

      if (tokenizer_add_space_prefix_) {
        if (text.length() > 0 && text[0] == ' ') {
          text = text.substr(1);
        }
      }

      if (case_encoding_ && text.back() == ' ') {
        text.pop_back();
      }
      decoded_strings.push_back(text);
    }

    std::unique_ptr<TokenizerDecodingState> decoding_state(state);
    output.SetStringOutput(decoded_strings, output_dim);
    return {};
  }

  // Helper: Decode first UTF-8 codepoint
  bool DecodeFirstUTF8Codepoint(const std::string& utf8, wchar_t& codepoint, size_t& char_len) const {
    unsigned char lead = static_cast<unsigned char>(utf8[0]);
    if (lead < 0x80) {
      codepoint = lead;
      char_len = 1;
    } else if ((lead >> 5) == 0x6) {
      if (utf8.size() < 2) return false;
      codepoint = ((lead & 0x1F) << 6) | (utf8[1] & 0x3F);
      char_len = 2;
    } else if ((lead >> 4) == 0xE) {
      if (utf8.size() < 3) return false;
      codepoint = ((lead & 0x0F) << 12) |
                  ((utf8[1] & 0x3F) << 6) |
                  (utf8[2] & 0x3F);
      char_len = 3;
    } else if ((lead >> 3) == 0x1E) {
      if (utf8.size() < 4) return false;
      codepoint = ((lead & 0x07) << 18) |
                  ((utf8[1] & 0x3F) << 12) |
                  ((utf8[2] & 0x3F) << 6) |
                  (utf8[3] & 0x3F);
      char_len = 4;
    } else {
      return false;
    }
    return true;
  }

  // Helper: Encode a wchar_t as UTF-8
  std::string EncodeUTF8(wchar_t wc) const {
    std::string out;

    // Promote wchar_t to uint32_t to avoid data loss from shift operations and silence warning C4333
    uint32_t u = static_cast<uint32_t>(wc);

    if (u < 0x80) {
      out += static_cast<char>(u);
    } else if (u < 0x800) {
      out += static_cast<char>((u >> 6) | 0xC0);
      out += static_cast<char>((u & 0x3F) | 0x80);
    } else if (u < 0x10000) {
      out += static_cast<char>((u >> 12) | 0xE0);
      out += static_cast<char>(((u >> 6) & 0x3F) | 0x80);
      out += static_cast<char>((u & 0x3F) | 0x80);
    } else {
      out += static_cast<char>((u >> 18) | 0xF0);
      out += static_cast<char>(((u >> 12) & 0x3F) | 0x80);
      out += static_cast<char>(((u >> 6) & 0x3F) | 0x80);
      out += static_cast<char>((u & 0x3F) | 0x80);
    }

    return out;
  }

  // Updated titlecase logic (basic toupper does not work for a lot of languages)
  void TitlecaseFirstCharacter(std::string& token) const {
    if (token.empty()) return;

    wchar_t codepoint;
    size_t char_len = 0;

    if (!DecodeFirstUTF8Codepoint(token, codepoint, char_len)) return;

    // Unicode-aware titlecasing for Cyrillic
    if (codepoint >= L'а' && codepoint <= L'я') {
      codepoint = codepoint - (L'а' - L'А');  // Convert to uppercase
    } else if (codepoint == L'ё') {
      codepoint = L'Ё';  // Special case
    } else {
      codepoint = std::towupper(codepoint);  // Fallback (Latin, etc.)
    }

    std::string prefix = EncodeUTF8(codepoint);
    std::string suffix = token.substr(char_len);
    token = prefix + suffix;
  }

  OrtxStatus Id2Token(extTokenId_t id, std::string& token, TokenizerDecodingState** state) const {
    std::unique_ptr<TokenizerDecodingState> decoding_state;
    if (*state == nullptr) {
      decoding_state = std::make_unique<TokenizerDecodingState>();
      *state = decoding_state.release();
    }

    if (special_token_ids_.count(id)) {
      token = "";
      return {};
    }

    if (id >= vocab_.size()) {
      token = unknown_token_;
      return {};
    }

    token = vocab_[id];
    if (case_encoding_ && token.length() == 1) {
      if (token[0] == normalizer::cUppercase || token[0] == normalizer::cAllUppercase ||
          token[0] == normalizer::cTitlecase || token[0] == normalizer::cLowercase ||
          token[0] == normalizer::cPunctuation) {
        (*state)->signature_ = token[0];
        token = "";
        return {};
      }
    }

    const std::string ws = " ";
    auto pos = token.find(spm_escaped_space);
    if (pos != std::string::npos) {
      if (pos == 0) {
        token = ws + token.substr(spm_escaped_space.length());
      } else if (pos + 3 == token.length()) {
        token = token.substr(0, pos) + ws;
      }
    }
    
    if (!case_encoding_) {
      return {};
    }

    char signature = 0;
    if ((*state)->signature_ != 0) {
      signature = (*state)->signature_;
      (*state)->signature_ = 0;
    }

    if (signature) {
      // Apply transformation from previous token's signature
      switch (signature) {
        case normalizer::cUppercase:
        case normalizer::cAllUppercase:
          std::transform(token.begin(), token.end(), token.begin(), ::toupper);
          break;
        case normalizer::cTitlecase:
          TitlecaseFirstCharacter(token);
          break;
        case normalizer::cLowercase:
        case normalizer::cPunctuation:
          // No transformation needed
          break;
      }
    } else if (!token.empty()) {
      // Check if current token starts with a signature character
      char first_char = token[0];
      if (first_char == normalizer::cUppercase || first_char == normalizer::cAllUppercase ||
          first_char == normalizer::cTitlecase || first_char == normalizer::cLowercase ||
          first_char == normalizer::cPunctuation) {
        token.erase(0, 1);  // Remove signature character

        switch (first_char) {
          case normalizer::cUppercase:
          case normalizer::cAllUppercase:
            std::transform(token.begin(), token.end(), token.begin(), ::toupper);
            break;
          case normalizer::cTitlecase:
            TitlecaseFirstCharacter(token);
            break;
            // For cLowercase and cPunctuation, no transformation needed
        }
      }
    }

    return {};
  }

 private:
  bool tokenizer_add_space_prefix_ = true;
  bool case_encoding_ = false;
  std::vector<std::string> vocab_;
  std::string unknown_token_ = "<unk>";
  std::set<extTokenId_t> special_token_ids_;
};

}  // namespace ort_extensions