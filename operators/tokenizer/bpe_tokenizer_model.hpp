// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "narrow.h"
#include "ustring.h"
#include "string_utils.h"
#include "string_tensor.h"

#include <list>
#include <unordered_map>
#include <iostream>
#include <utility>
#include <charconv>
#include <limits>

#include "nlohmann/json.hpp"
#include "bpe_utils.hpp"
#include "trietree.hpp"
#include "tokenizer_common.h"

namespace ort_extensions {

class BpeModel {
  using json = nlohmann::json;

 public:
  BpeModel() = default;

  static void UpdateSpmByteToken(std::unordered_map<std::string, uint32_t>& vocab_map) {
    static const char* hex = "0123456789ABCDEF";
    for (char32_t ch = 0; ch < 256; ++ch) {
      std::string tok(1, ort_extensions::narrow<unsigned char>(ch));
      if (vocab_map.find(tok) != vocab_map.end()) {
        continue;
      }

      const char buf[7] = {'<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0};
      auto it = vocab_map.find(buf);
      if (it != vocab_map.end()) {
        vocab_map[tok] = it->second;
      }
    }
  }

  OrtxStatus Load(std::istream& vocab_stream, std::istream& merges_stream, const char* unk_token,
                  const char* special_tokens, bool spm_converted) {
    nlohmann::json tok_json;
    vocab_stream >> tok_json;
    tok_json.get_to(vocab_map_);

    auto it = vocab_map_.find(unk_token);
    if (it != vocab_map_.end()) {
      unk_id_ = it->second;
    } else {
      auto id = ort_extensions::narrow<uint32_t>(vocab_map_.size());
      vocab_map_[unk_token] = id;
      unk_id_ = id;
    }

    if (spm_converted) {
      UpdateSpmByteToken(vocab_map_);
    }

    uint32_t index = 0;
    std::string line;
    while (std::getline(merges_stream, line)) {
      line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
      if (line.empty()) continue;
      if ((line[0] == '#') && (index == 0)) continue;
      auto pos = line.find(' ');
      if (pos == std::string::npos) {
        return {
            kOrtxErrorCorruptData,
            "Cannot know how to parse line: " + line,
        };
      }
      std::string w1 = line.substr(0, pos);
      std::string w2 = line.substr(pos + 1);
      int token_length = ort_extensions::narrow<int>(w1.length() + w2.length());
      if (w2.find("</w>") != std::string::npos || w1.find("</w>") != std::string::npos) {
        token_length -= 4;
      }
      auto iw1 = GetTokenId(w1);
      auto iw2 = GetTokenId(w2);
      auto iww = GetTokenId(w1 + w2);
      BpeNode value{iww, index++, token_length};
      bpe_rank_[GetRankKey(iw1, iw2)] = value;
    }

    if (special_tokens != nullptr) {
      std::istringstream istrea(special_tokens);

      while (istrea >> line) {
        if (line.empty()) continue;
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        ustring line_32(line);
        auto id = ort_extensions::narrow<uint32_t>(vocab_map_.size());
        if (auto nestedIt = vocab_map_.find(line); nestedIt != vocab_map_.end()) {
          id = nestedIt->second;
        } else {
          vocab_map_[line] = id;
        }
        special_tokens_.Add(std::move(line_32), id);
      }
    }

    id2token_map_.resize(vocab_map_.size());
    for (const auto& [t, i] : vocab_map_) {
      if (i > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)())) {
        continue;  // safe purpose.
      }
      if (i > id2token_map_.size()) {
        id2token_map_.resize(static_cast<size_t>(i) + 1);
      }
      id2token_map_[i] = t;
    }

    return {};
  }

  OrtxStatus Load(const json& bpe_model, const char* /* special_tokens */, bool spm_converted) {
    const json& vocab_json = bpe_model["vocab"];
    const json& merges_json = bpe_model["merges"];
    vocab_json.get_to(vocab_map_);
    auto it = bpe_model.find("unk_token");
    if (it != bpe_model.end() && it->is_string()) {
      auto ukt = it->get<std::string>();
      auto it_word = vocab_map_.find(ukt);
      if (it_word != vocab_map_.end()) {
        unk_id_ = it_word->second;
      }
    }

    it = bpe_model.find("end_of_word_suffix");
    if (it != bpe_model.end() && it->is_string()) {
      end_of_word_suffix_ = it->get<std::string>();
    }

    if (spm_converted) {
      UpdateSpmByteToken(vocab_map_);
    }

    uint32_t index = 0;
    auto merge_item = merges_json.begin();
    while (merge_item != merges_json.end()) {
      std::string w1, w2;
      if (merge_item->is_string()) {
        std::string line = merge_item.value();
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        if (line.empty()) continue;
        if ((line[0] == '#') && (index == 0)) continue;
        auto pos = line.find(' ');
        if (pos == std::string::npos) {
          return {
              kOrtxErrorCorruptData,
              "Cannot know how to parse line: " + line,
          };
        }
        w1 = line.substr(0, pos);
        w2 = line.substr(pos + 1);
      } else if (merge_item->is_array()) {
        w1 = merge_item->at(0).get<std::string>();
        w2 = merge_item->at(1).get<std::string>();
      } else {
        return {kOrtxErrorCorruptData, "Cannot know how to parse line: " + merge_item->dump()};
      }
      int token_length = ort_extensions::narrow<int>(w1.length() + w2.length());
      if (w2.find("</w>") != std::string::npos || w1.find("</w>") != std::string::npos) {
        token_length -= 4;
      }

      auto iw1 = GetTokenId(w1);
      auto iw2 = GetTokenId(w2);
      auto iww = GetTokenId(w1 + w2);
      BpeNode value{iww, index++, token_length};
      bpe_rank_[GetRankKey(iw1, iw2)] = value;

      merge_item++;
    }

    id2token_map_.resize(vocab_map_.size());
    for (const auto& [t, i] : vocab_map_) {
      if (i > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)())) {
        continue;  // safe purpose.
      }
      if (i > id2token_map_.size()) {
        id2token_map_.resize(static_cast<size_t>(i) + 1);
      }
      id2token_map_[i] = t;
    }

    return {};
  }

  OrtxStatus Load(std::unordered_map<std::string, uint32_t>& vocab,
                  std::vector<std::pair<std::string, std::string>>& merges, const char* /* special_tokens */,
                  bool spm_converted) {
    vocab_map_ = vocab;

    if (spm_converted) {
      UpdateSpmByteToken(vocab_map_);
    }

    uint32_t index = 0;
    for (auto& tuple : merges) {
      std::string w1 = tuple.first;
      std::string w2 = tuple.second;
      int token_length = ort_extensions::narrow<int>(w1.length() + w2.length());
      if (w2.find("</w>") != std::string::npos || w1.find("</w>") != std::string::npos) {
        token_length -= 4;
      }
      auto iw1 = GetTokenId(w1);
      auto iw2 = GetTokenId(w2);
      auto iww = GetTokenId(w1 + w2);
      BpeNode value{iww, index++, token_length};
      bpe_rank_[GetRankKey(iw1, iw2)] = value;
    }

    id2token_map_.resize(vocab_map_.size());
    for (const auto& [t, i] : vocab_map_) {
      if (i > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)())) {
        continue;  // safe purpose.
      }
      if (i > id2token_map_.size()) {
        id2token_map_.resize(static_cast<size_t>(i) + 1);
      }
      id2token_map_[i] = t;
    }

    return {};
  }

  OrtxStatus LoadAddedTokens(const char* added_tokens) {
    int id = bpe::kInvalidTokenId;
    std::istringstream strm_tokens(added_tokens);
    std::string line;
    while (!strm_tokens.eof()) {
      std::getline(strm_tokens, line);
      line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
      if (line.empty()) continue;
      // separate the key and value by =
      auto pos = line.rfind("=");
      if (pos == std::string::npos) {
        return {kOrtxErrorCorruptData, "Error on parse a added_token line: " + line};
      }
      auto token = line.substr(0, pos);
      auto id_str = line.substr(pos + 1);  // 1 is the length of "="
      auto [ptr, ec] = std::from_chars(id_str.data(), id_str.data() + id_str.length(), id);
      if (ec != std::errc()) {
        return {kOrtxErrorCorruptData, "Cannot convert to an integer from " + id_str};
      }

      added_tokens_.Add(ustring(token), 0, std::make_optional(id));
    }

    return {};
  }

  void LoadAddedTokens(const std::vector<AddedToken>& added_tokens) {
    for (const auto& token : added_tokens) {
      added_tokens_.Add(ustring(token.content_), 0, token.id_);
    }
  }

  std::vector<std::string> BuildDecoder() const { return id2token_map_; }

  // REF:
  // https://github.com/huggingface/transformers/blob/c9e72f55b2dc4b9be4edb986dce0552582b328f2/src/transformers/tokenization_utils.py#L52
  bpe::TokenPairs SplitByAddedAndSpecial(const ustring& input) const {
    // split by added tokens
    bpe::TokenPairs added_result;
    bpe::TokenPairs final_result;
    added_tokens_.Split(input, added_result);
    for (const auto& [token, id] : added_result) {
      if (id != bpe::kInvalidTokenId) {
        final_result.emplace_back(token, id);
      } else {
        auto special_result = special_tokens_.SplitBySpecialTokens(token);
        for (const auto& [token, id] : special_result) {
          final_result.emplace_back(token, id);
        }
      }
    }

    return final_result;
  }

  void PerformBPE(std::list<std::pair<uint32_t, uint32_t>>& vals) const {
    while (vals.size() >= 2) {
      auto pos_it = vals.end();
      uint32_t minval = (std::numeric_limits<uint32_t>::max)();
      uint32_t ori_id1 = 0, ori_id2 = 0;
      uint32_t aim_id = 0;
      int token_length = 0;
      for (auto it = vals.begin(); it != vals.end(); ++it) {
        auto it2 = it;
        ++it2;
        if (it2 == vals.end()) {
          break;
        }

        auto map_it = bpe_rank_.find(GetRankKey(it->first, it2->first));
        if (map_it == bpe_rank_.end()) {
          continue;
        }

        if (minval > map_it->second.value) {
          ori_id1 = it->first;
          ori_id2 = it2->first;
          minval = map_it->second.value;
          pos_it = it;
          aim_id = map_it->second.id;
        }
      }

      if (pos_it == vals.end()) {
        break;
      }

      token_length = pos_it->second;
      pos_it = vals.erase(pos_it);
      pos_it->first = aim_id;
      pos_it->second += token_length;
      for (++pos_it; pos_it != vals.end(); ++pos_it) {
        if (pos_it->first != ori_id1) continue;
        auto it2 = pos_it;
        ++it2;
        if (it2 == vals.end()) break;
        if (it2->first != ori_id2) continue;
        token_length = pos_it->second;
        pos_it = vals.erase(pos_it);
        pos_it->first = aim_id;
        pos_it->second += token_length;
      }
    }
  }

  uint32_t GetTokenId(const std::string& key) const {
    auto it = vocab_map_.find(key);
    if (it != vocab_map_.end()) {
      return it->second;
    } else {
      return bpe::kInvalidTokenId;
    }
  }

  uint32_t GetAddedTokenId(const std::string& key) const {
    size_t idx = 0;
    int id = added_tokens_.FindLongest(ustring(key), idx);
    if (idx == 0) {
      return bpe::kInvalidTokenId;
    }

    return static_cast<uint32_t>(id);
  }

  const std::string& GetEndOfWordSuffix() const { return end_of_word_suffix_; }

 private:
  struct BpeNode {
    uint32_t id;
    uint32_t value;
    int length;
  };

  static uint64_t GetRankKey(uint32_t i0, uint32_t i1) {
    return (static_cast<uint64_t>(i1) << 32) | (i0 & 0xFFFFFFFFLL);
  }

 private:
  std::string end_of_word_suffix_;
  std::map<uint64_t, BpeNode> bpe_rank_;

  std::unordered_map<std::string, uint32_t> vocab_map_;
  std::vector<std::string> id2token_map_;

  uint32_t unk_id_ = (std::numeric_limits<uint32_t>::max)();
  bpe::SpecialTokenMap special_tokens_;
  TrieTree<char32_t> added_tokens_;
};

}  // namespace ort_extensions
