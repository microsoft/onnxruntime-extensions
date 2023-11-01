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

#include "nlohmann/json.hpp"
#include "bpe_utils.hpp"
#include "trietree.hpp"

class BpeModel {
 public:
  BpeModel() = default;

  OrtStatusPtr LoadAddedTokens(const char* added_tokens) {
    std::istringstream istrea(added_tokens);
    std::string line;
    while (istrea >> line) {
      if (line.empty()) continue;
      line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
      add_tokens_.Add(ustring(line));
    }

    return nullptr;
  }

  OrtStatusPtr Load(std::istream& vocab_stream,
                    std::istream& merges_stream,
                    const char* unk_token,
                    const char* special_tokens) {
    nlohmann::json tok_json;
    vocab_stream >> tok_json;
    vocab_map_ = std::move(tok_json.get<std::unordered_map<std::string, uint32_t>>());

    auto it = vocab_map_.find(unk_token);
    if (it != vocab_map_.end()) {
      unk_id_ = it->second;
    } else {
      auto id = ort_extensions::narrow<uint32_t>(vocab_map_.size());
      vocab_map_[unk_token] = id;
      unk_id_ = id;
    }

    CreateByteEncoder();

    uint32_t index = 0;
    std::string line;
    while (std::getline(merges_stream, line)) {
      line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
      if (line.empty()) continue;
      if ((line[0] == '#') && (index == 0)) continue;
      auto pos = line.find(' ');
      if (pos == std::string::npos) {
        return OrtW::CreateStatus("Cannot know how to parse line: " + line, ORT_INVALID_ARGUMENT);
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
      id2token_map_[i] = t;
    }

    return nullptr;
  }

  void bpe(std::list<std::pair<uint32_t, uint32_t>>& vals) const {
    while (vals.size() >= 2) {
      auto pos_it = vals.end();
      uint32_t minval = std::numeric_limits<uint32_t>::max();
      uint32_t ori_id1 = 0, ori_id2 = 0;
      uint32_t aim_id = 0;
      int token_length = 0;
      for (auto it = vals.begin(); it != vals.end(); ++it) {
        auto it2 = it;
        ++it2;
        if (it2 == vals.end()) break;
        auto map_it = bpe_rank_.find(GetRankKey(it->first, it2->first));
        if (map_it == bpe_rank_.end()) continue;
        if (minval > map_it->second.value) {
          ori_id1 = it->first;
          ori_id2 = it2->first;
          minval = map_it->second.value;
          pos_it = it;
          aim_id = map_it->second.id;
        }
      }
      if (pos_it == vals.end()) break;

      token_length = pos_it->second;
      pos_it = vals.erase(pos_it);
      pos_it->first = aim_id;
      pos_it->second = pos_it->second + token_length;
      for (++pos_it; pos_it != vals.end(); ++pos_it) {
        if (pos_it->first != ori_id1) continue;
        auto it2 = pos_it;
        ++it2;
        if (it2 == vals.end()) break;
        if (it2->first != ori_id2) continue;
        token_length = pos_it->second;
        pos_it = vals.erase(pos_it);
        pos_it->first = aim_id;
        pos_it->second = pos_it->second + token_length;
      }
    }
  }

  const auto& ByteEncoder() const {
    return byte_encoder_;
  }

  // REF: https://github.com/huggingface/transformers/blob/7d8ff3629b2725ec43ace99c1a6e87ac1978d433/src/transformers/tokenization_utils_base.py#L82
  // https://github.com/Narsil/transformers/blob/d6e64d3c2396cc3cd095778446cf6bae9495c8f2/src/transformers/tokenization_utils.py#L90
  auto SplitByAddedTokens(const ustring& input) const {
    size_t offset = 0;
    std::vector<std::u32string_view> tokens;
    while (offset < input.length()) {
      auto token = add_tokens_.FindLongest(input, offset);
      if (token == 0) {
        offset += 1;
      } else {
        offset += 1;
        yield(token);
      }

    }
    return tokens;
  }

  auto SplitBySpecialTokens(const ustring& input) const {
    return special_tokens_.SplitBySpecialTokens(input);
  }

  // Returns token if key was found in vocab, and unk_id_ otherwise
  uint32_t GetTokenId(const std::string& key) {
    auto it = vocab_map_.find(key);
    if (it != end(vocab_map_)) {
      return it->second;
    } else {
      return unk_id_;
    }
  }

 private:
  struct BpeNode {
    uint32_t id;
    uint32_t value;
    int length;
  };

  static uint64_t GetRankKey(uint32_t i0, uint32_t i1) {
    return (static_cast<uint64_t>(i1) << 32) | (i0 & 0xFFFFFFFFLL);
  }

  void CreateByteEncoder() {
    char32_t index = 256;
    for (char32_t i = 0; i < 256; ++i) {
      /*
      bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
      )
      */
      if ((i >= 0 && i < 33) || (i >= 127 && i < 161) || (i == 173)) {
        byte_encoder_[i] = GetTokenId(ustring::EncodeUTF8Char(index++));
      } else {
        byte_encoder_[i] = GetTokenId(ustring::EncodeUTF8Char(i));
      }
    }
  }

 private:
  std::map<uint64_t, BpeNode> bpe_rank_;

  uint32_t byte_encoder_[256] = {};
  std::unordered_map<std::string, uint32_t> vocab_map_;
  std::vector<std::string> id2token_map_;

  uint32_t unk_id_ = std::numeric_limits<uint32_t>::max();
  SpecialTokenMap special_tokens_;
  TrieTree<char32_t> add_tokens_;
};
