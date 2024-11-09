// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// The implementation is inspired by llama.cpp ugm tokenizer and huggingface FastTokenizer

#pragma once

#include <map>
#include <set>
#include <list>
#include <string>
#include <vector>
#include <cfloat>
#include <functional>
#include <unordered_map>

#include "ortx_tokenizer.h"
#include "ext_status.h"
#include "op_def_struct.h"
#include "base64.h"
#include "ustring.h"
#include "narrow.h"
#include "nlohmann/json.hpp"
#include "trietree.hpp"
#include "tokenizer_jsconfig.hpp"

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
    if (normalizer != j_vocab.end()) {
      auto iter = normalizer->find("precompiled_charsmap");
      if (iter != normalizer->end()) {
        auto charsmap = iter->get<std::string>();
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

    extTokenId_t id = 0;
    for (const auto& entry : vocab_node->items()) {
      auto tkn = entry.value()[0].get<std::string>();
      auto score = entry.value()[1].get<double>();
      vocab_[tkn] = std::make_tuple(id++, score);
    }

    scores_.resize(id);
    double min_score = -DBL_MAX;
    for (const auto& entry : vocab_) {
      scores_[std::get<0>(entry.second)] = std::get<1>(entry.second);
      token_matcher_.Add(entry.first, 0, std::get<0>(entry.second));
      min_score = std::min<double>(min_score, std::get<1>(entry.second));
    }

    unknown_token_score_ = min_score - unknown_token_score_penalty_;
    return status;
  }

  extTokenId_t GetTokenId(const std::string& token) const { 
    auto iter = vocab_.find(token);
    if (iter == vocab_.end()) {
      return special_unk_id_;
    }
    return std::get<0>(iter->second);
  }

  OrtxStatus Compute(const ortc::Tensor<std::string>& input, ortc::Tensor<int64_t>& tokenize_output) const {
    if (input.Shape().size() != 1) {
      return OrtxStatus(extError_t::kOrtxErrorInvalidArgument, "Input tensor must have rank 1.");
    }

    std::string normalized;
    Normalize(input.AsScalar(), &normalized);
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
        const double challenger_score = current_best.score_sum + unknown_token_score_;
        prefix_offset = input_offset + n_utf8_code_units;
        struct BestTokenization& current_champ = tokenization_results[prefix_offset];
        if (challenger_score > current_champ.score_sum) {
          struct BestTokenization challenger = {input_offset, (float)challenger_score, special_unk_id_};
          current_champ = challenger;
        }
      }

      input_offset += n_utf8_code_units;
    }

    std::vector<extTokenId_t> output;
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

    bool add_bos = GetTokenId(bos_token_) != special_unk_id_;
    bool add_eos = GetTokenId(eos_token_) != special_unk_id_;
    auto output_size = static_cast<int64_t>(output.size());
    int64_t* id_output = tokenize_output.Allocate({output_size + add_bos + add_eos});
    if (add_bos) {
      *id_output = GetTokenId(bos_token_);
      id_output++;
    }
    std::transform(output.begin(), output.end(), id_output, [](extTokenId_t id) { return static_cast<int64_t>(id); });
    std::reverse(id_output, id_output + output_size);
    if (add_eos) {
      *(id_output + output_size) = GetTokenId(eos_token_);
    }
    return {};
  }

 private:
  struct NormalizationResult {
    const char* normalized;
    size_t normalized_len;
    size_t consumed_input;
  };

  void Normalize(const std::string& input, std::string* normalized) const {
    normalized->clear();
    normalized->reserve(input.size() * 3);

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
              normalized->append(space);
              is_space_prepended = true;
            }
          }
          normalized->push_back(c);
        } else {
          if (processing_non_ws) {
            processing_non_ws = false;
          }
          if (!shall_merge_spaces) {
            normalized->append(space);
          }
        }
      }

      input_offset += norm_res.consumed_input;
    }

    if (shall_append_space) {
      normalized->append(space);
    }
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

  struct NormalizationResult NormalizePrefix(const std::string& input, size_t input_offset) const {
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
      } else {
        return {"\xEF\xBF\xBD", 3, 1};
      }
    }
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
  double unknown_token_score_;

  Vocab vocab_;
  std::vector<double> scores_;
  std::set<extTokenId_t> special_token_ids_;
  VocabTrieTree token_matcher_;

 public:
  bool tokenizer_escape_whitespaces_ = true;
  bool tokenizer_treat_whitespace_as_suffix_ = false;
  bool tokenizer_add_space_prefix_ = true;
  bool tokenizer_remove_extra_whitespaces_ = true;
  std::string bos_token_ = "<s>";
  std::string eos_token_ = "</s>";
  std::string pad_token_ = "<pad>";
  std::string unk_token_ = "<unk>";
};


class SpmUgmDecoder {
 public:
  SpmUgmDecoder() {
  }

  OrtxStatus Load(const TokenJsonConfig& config, const SpmUgmTokenizer& tokenizer) {
    auto vocab_size = tokenizer.vocab_.size();
    vocab_.resize(vocab_size);
    for (auto iter = tokenizer.vocab_.begin(); iter != tokenizer.vocab_.end(); ++iter) {
      vocab_[std::get<0>(iter->second)] = iter->first;
    }

    unknown_token_ = tokenizer.unk_token_;
    special_token_ids_ = tokenizer.special_token_ids_;
    tokenizer_add_space_prefix_ = tokenizer.tokenizer_add_space_prefix_;
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
    const std::string ws = " ";
    for (auto n = string_batch; n > 0; n--) {
      std::string text;
      for (int64_t i = 0; i < seq_len; ++i) {
        std::string token;
        Id2Token(ort_extensions::narrow<extTokenId_t>(p_ids[i]), token, nullptr);
        if (token.find(spm_escaped_space) == 0) {
          token = ws + token.substr(spm_escaped_space.length());
        }

        text += token;
      }

      if (tokenizer_add_space_prefix_) {
        if (text.length() > 0 && text[0] == ' ') {
          text = text.substr(1);
        }
      }

      decoded_strings.push_back(text);
    }

    output.SetStringOutput(decoded_strings, output_dim);
    return {};
  }

  OrtxStatus Id2Token(extTokenId_t id, std::string& token, TokenizerDecodingState** /* state */) const {
    if (special_token_ids_.count(id)) {
      token = "";
      return {};
    }

    if (id >= vocab_.size()) {
      token = unknown_token_;
      return {};
    }

    token = vocab_[id];
    return {};
  }

private:
  bool tokenizer_add_space_prefix_ = true;
  std::vector<std::string> vocab_;
  std::string unknown_token_ = "<unk>";
  std::set<extTokenId_t> special_token_ids_;
};

}  // namespace ort_extensions
