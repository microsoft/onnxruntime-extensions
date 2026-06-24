// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "narrow.h"
#include "ustring.h"
#include "string_utils.h"
#include "string_tensor.h"

#include <set>
#include <list>
#include <vector>
#include <queue>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <utility>
#include <charconv>
#include <limits>

#include "nlohmann/json.hpp"
#include "bpe_utils.hpp"
#include "trietree.hpp"
#include "tokenizer_common.h"

#define ORTX_JSON_RETURN_IF_NULL(node_iter, name, var) \
  auto var = (node_iter)->find(name);                  \
  if (var == (node_iter)->end() || var->is_null()) {   \
    return {};                                         \
  }

namespace ort_extensions {

class BpeModel {
  using json = nlohmann::json;
  const std::array<const char*, 12> kPreTokenizerType = {
      "BertPreTokenizer", "ByteLevel",       "CharDelimiterSplit", "Digits", "Metaspace",
      "PreTokenizer",     "Punctuation",     "Sequence",           "Split",  "UnicodeScripts",
      "Whitespace",       "WhitespaceSplit",
  };

 public:
  BpeModel()
      : pre_tokenizer_types_(kPreTokenizerType.begin(), kPreTokenizerType.end()) {};

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

  OrtxStatus LoadPreTokenizer(const json& bpe_model) {
    auto root_node = &bpe_model;
    ORTX_JSON_RETURN_IF_NULL(root_node, "pre_tokenizer", node_pre_tokenizer);
    auto iter_type = node_pre_tokenizer->find("type");
    if (iter_type != node_pre_tokenizer->end() && !iter_type->is_null()) {
      auto pre_token_type = iter_type->get<std::string>();
      if (pre_tokenizer_types_.count(pre_token_type) == 0) {
        return {kOrtxErrorNotImplemented, std::string("Unsupported pretokenizer type!") + pre_token_type};
      }

      // Handle top-level Split pre-tokenizer (not nested in a Sequence/pretokenizers array).
      // E.g., chatglm3 has: {"type": "Split", "pattern": {"String": "<!dummy-prefix!>"}}
      // A Split on a literal String that never appears in normal text is effectively a no-op.
      if (pre_token_type == "Split") {
        auto iter_pattern = node_pre_tokenizer->find("pattern");
        if (iter_pattern != node_pre_tokenizer->end()) {
          auto iter_regex = iter_pattern->find("Regex");
          if (iter_regex != iter_pattern->end() && !iter_regex->is_null()) {
            // Split with a Regex pattern — use it as the pre-tokenizer regex
            pre_tokenizer_regex_ = iter_regex->get<std::string>();
            bpe::PreTokenizerWithRegEx pre_tokenizer;
            auto status = pre_tokenizer.Compile(pre_tokenizer_regex_);
            if (!status.IsOk()) {
              return status;
            }
          } else {
            // Split with a String pattern (no-op for tokenization — never matches normal text)
            no_op_pretokenizer_ = true;
          }
        }
        return {};
      }
    }

    ORTX_JSON_RETURN_IF_NULL(node_pre_tokenizer, "pretokenizers", iter_node_list);

    for (const auto& node : *iter_node_list) {
      ORTX_JSON_RETURN_IF_NULL(&node, "type", iter_type);
      auto pre_type = iter_type->get<std::string>();
      if (pre_type == "Split") {
        ORTX_JSON_RETURN_IF_NULL(&node, "pattern", iter_pattern);
        ORTX_JSON_RETURN_IF_NULL(iter_pattern, "Regex", regex_str);
        auto regex = NormalizeJsonRegexEscapes(regex_str->get<std::string>());
        bpe::PreTokenizerWithRegEx pre_tokenizer;
        auto status = pre_tokenizer.Compile(regex);
        if (!status.IsOk()) {
          return status;
        }
        sequence_steps_.push_back({std::move(regex)});
      } else {
        if (pre_tokenizer_types_.count(pre_type) == 0) {
          return {kOrtxErrorNotImplemented, "Unsupported pretokenizer type!"};
        }
      }
    }

    return {};
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
        ORTX_RETURN_IF_ERROR(special_tokens_.Add(std::move(line_32), id));
      }
    }

    id2token_map_.resize(vocab_map_.size());
    for (const auto& [t, i] : vocab_map_) {
      if (i > static_cast<uint32_t>((std::numeric_limits<int32_t>::max)())) {
        continue;  // safe purpose.
      }
      if (i >= id2token_map_.size()) {
        id2token_map_.resize(static_cast<size_t>(i) + 1);
      }
      id2token_map_[i] = t;
    }

    BuildSpeculativeTrie();
    return {};
  }

  OrtxStatus Load(const json& bpe_model, const json& tok_json, const char* /* special_tokens */, bool spm_converted) {
    ORTX_RETURN_IF_ERROR(LoadPreTokenizer(tok_json));

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
      if (i >= id2token_map_.size()) {
        id2token_map_.resize(static_cast<size_t>(i) + 1);
      }
      id2token_map_[i] = t;
    }

    BuildSpeculativeTrie();
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
      if (i >= id2token_map_.size()) {
        id2token_map_.resize(static_cast<size_t>(i) + 1);
      }
      id2token_map_[i] = t;
    }

    BuildSpeculativeTrie();
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

  void LoadAddedTokens(const AddedTokenMap& added_tokens) {
    for (const auto& [key, token] : added_tokens) {
      added_tokens_.Add(ustring(token.content_), 0, token.id_);
    }
  }

  std::vector<std::string> BuildDecoder() const { return id2token_map_; }

  // REF:
  // https://github.com/huggingface/transformers/blob/c9e72f55b2dc4b9be4edb986dce0552582b328f2/src/transformers/tokenization_utils.py#L52
  bpe::TokenPairs SplitByAddedAndSpecial(const ustring& input, const AddedTokenMap& t_map) const {
    static const std::set<char32_t> ws_chars = {U' ', U'\n', U'\r', U'\t'};
    // split by added tokens
    bpe::TokenPairs added_result;
    bpe::TokenPairs final_result;
    added_tokens_.Split(input, added_result);

    for (size_t n = 0; n < added_result.size(); ++n) {
      auto& [token, id] = added_result[n];
      bool has_left = n > 0;
      bool has_right = n < added_result.size() - 1;

      if (id != bpe::kInvalidTokenId) {
        if (has_left || has_right) {
          auto iter_tok_extend = t_map.find(std::u32string(token));
          if (iter_tok_extend != t_map.end()) {
            if (has_right && iter_tok_extend->second.rstrip_) {
              auto& [next_token, next_id] = added_result[n + 1];
              // r-strip removes trailing characters from right side, which is equivalent to removing whitespace from left side of next token
              if (next_id == bpe::kInvalidTokenId) {
                final_result.emplace_back(token, id);
                size_t pos = 0;
                while (pos < next_token.size() && ws_chars.count(next_token[pos])) {
                  pos++;
                }
                auto stripped_token = next_token.substr(pos);
                final_result.emplace_back(stripped_token, next_id);
                n += 1;
                continue;
              }
            }
            if (has_left && iter_tok_extend->second.lstrip_) {
              auto& [prev_token, prev_id] = added_result[n - 1];
              // l-strip means remove whitespaces from right side of previous token
              if (prev_id == bpe::kInvalidTokenId) {
                size_t pos = token.size();
                while (pos > 0 && ws_chars.count(token[pos - 1])) {
                  pos--;
                }
                auto stripped_token = token.substr(0, pos);
                final_result.emplace_back(token, id);
                continue;
              }
            }
          }
        }
        // if not additional processing, just add it to final result
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

  void PerformBPE(std::vector<std::pair<uint32_t, uint32_t>>& vals) const {
    if (vals.size() < 2) return;

    // TODO(Phase 2): Speculative O(N) greedy tokenization.
    // Currently disabled — BPE-consistency verification rejects valid tokens
    // for models with overlapping merge paths (Gemma, LLaMA SPM, ChatGLM).
    // Need to rethink the consistency check before enabling.
    // if (spec_trie_valid_ && PerformBPE_Speculative(vals)) {
    //   return;
    // }

    // Use appropriate algorithm based on sequence length.
    // For longer sequences, use priority-queue BPE: O(N log N) vs O(N*R) for flat scan.
    if (vals.size() > 32) {
      PerformBPE_Heap(vals);
      return;
    }

    PerformBPE_FlatScan(vals);
  }

 private:
  // O(N*R) flat-array scan: best for short sequences (pre-tokens from regex splitting).
  // R = number of distinct merge rounds needed.
  void PerformBPE_FlatScan(std::vector<std::pair<uint32_t, uint32_t>>& vals) const {
    // Flat-array linked list for cache-friendly traversal.
    // Each node stores token_id, byte_length, and prev/next indices.
    // Nodes are indexed [0..n-1]; next == -1 marks the end of the list.
    struct BpeListNode {
      uint32_t token_id;
      uint32_t byte_len;
      int32_t prev;  // -1 = no prev (head)
      int32_t next;  // -1 = no next (tail)
    };

    const size_t n = vals.size();
    // Stack buffer for common case (pre-tokens up to 128 chars); heap fallback for longer.
    constexpr size_t kStackSize = 128;
    BpeListNode stack_buf[kStackSize];
    std::unique_ptr<BpeListNode[]> heap_buf;
    BpeListNode* nodes = stack_buf;
    if (n > kStackSize) {
      heap_buf = std::make_unique<BpeListNode[]>(n);
      nodes = heap_buf.get();
    }

    // Initialize from vector (contiguous access, no linked-list traversal)
    for (size_t idx = 0; idx < n; idx++) {
      nodes[idx] = {vals[idx].first, vals[idx].second, static_cast<int32_t>(idx) - 1, static_cast<int32_t>(idx) + 1};
    }
    nodes[n - 1].next = -1;  // last node has no next

    size_t active_count = n;

    while (active_count >= 2) {
      // Find the pair with minimum merge rank
      int32_t best_idx = -1;
      uint32_t minval = (std::numeric_limits<uint32_t>::max)();
      uint32_t ori_id1 = 0, ori_id2 = 0;
      uint32_t aim_id = 0;

      for (int32_t i = 0; i != -1; i = nodes[i].next) {
        int32_t j = nodes[i].next;
        if (j == -1) break;

        auto map_it = bpe_rank_.find(GetRankKey(nodes[i].token_id, nodes[j].token_id));
        if (map_it == bpe_rank_.end()) continue;

        if (minval > map_it->second.value) {
          ori_id1 = nodes[i].token_id;
          ori_id2 = nodes[j].token_id;
          minval = map_it->second.value;
          best_idx = i;
          aim_id = map_it->second.id;
        }
      }

      if (best_idx == -1) break;

      // Merge all occurrences of (ori_id1, ori_id2) in one pass
      for (int32_t i = best_idx; i != -1;) {
        int32_t j = nodes[i].next;
        if (j == -1) break;

        if (nodes[i].token_id == ori_id1 && nodes[j].token_id == ori_id2) {
          // Merge: replace node i's token with merged token, absorb j's byte length, remove j
          nodes[i].token_id = aim_id;
          nodes[i].byte_len += nodes[j].byte_len;

          // Unlink j
          int32_t after_j = nodes[j].next;
          nodes[i].next = after_j;
          if (after_j != -1) {
            nodes[after_j].prev = i;
          }
          active_count--;

          // Continue from i (the merged node might form a new pair with its new next)
          // but don't advance — check i again with its new next
          continue;
        }
        i = nodes[i].next;
      }
    }

    // Write back to list
    vals.clear();
    for (int32_t i = 0; i != -1; i = nodes[i].next) {
      vals.emplace_back(nodes[i].token_id, nodes[i].byte_len);
    }
  }

  // O(N log N) priority-queue BPE: best for longer sequences where the flat scan
  // would do many rounds (e.g., Gemma with 256K merges and 100+ char pre-tokens).
  // Uses a min-heap of adjacent pairs keyed by merge rank with lazy deletion.
  // Based on Zouhar et al. "A Formal Perspective on Byte-Pair Encoding" (ACL 2023).
  void PerformBPE_Heap(std::vector<std::pair<uint32_t, uint32_t>>& vals) const {
    struct BpeListNode {
      uint32_t token_id;
      uint32_t byte_len;
      int32_t prev;
      int32_t next;
      bool removed;
    };

    const size_t n = vals.size();
    std::vector<BpeListNode> nodes(n);
    for (size_t i = 0; i < n; i++) {
      nodes[i] = {vals[i].first, vals[i].second, static_cast<int32_t>(i) - 1,
                  static_cast<int32_t>(i) + 1, false};
    }
    nodes[n - 1].next = -1;

    // Heap entry: merge rank (priority), resulting token, position (left node index),
    // and the token IDs when this entry was created (for lazy validation).
    struct HeapEntry {
      uint32_t rank;
      uint32_t merged_id;
      int32_t pos;         // index of left node
      uint32_t left_tok;   // token_id of left node at insertion time
      uint32_t right_tok;  // token_id of right node at insertion time

      bool operator>(const HeapEntry& o) const { return rank > o.rank; }
    };

    // Min-heap: lowest rank = highest priority
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;

    // Seed heap with all valid adjacent pairs
    for (size_t i = 0; i + 1 < n; i++) {
      auto it = bpe_rank_.find(GetRankKey(nodes[i].token_id, nodes[i + 1].token_id));
      if (it != bpe_rank_.end()) {
        heap.push({it->second.value, it->second.id, static_cast<int32_t>(i),
                   nodes[i].token_id, nodes[i + 1].token_id});
      }
    }

    while (!heap.empty()) {
      auto top = heap.top();
      heap.pop();

      // Lazy validation: check this entry is still valid
      int32_t i = top.pos;
      if (nodes[i].removed) continue;
      if (nodes[i].token_id != top.left_tok) continue;
      int32_t j = nodes[i].next;
      if (j == -1 || nodes[j].removed) continue;
      if (nodes[j].token_id != top.right_tok) continue;

      // Merge: keep node i, absorb node j
      nodes[i].token_id = top.merged_id;
      nodes[i].byte_len += nodes[j].byte_len;

      // Remove node j from linked list
      nodes[j].removed = true;
      int32_t after_j = nodes[j].next;
      nodes[i].next = after_j;
      if (after_j != -1) {
        nodes[after_j].prev = i;
      }

      // Add new adjacent pairs to heap
      // Left pair: (prev, i) — if prev exists
      if (nodes[i].prev != -1) {
        int32_t p = nodes[i].prev;
        auto it = bpe_rank_.find(GetRankKey(nodes[p].token_id, nodes[i].token_id));
        if (it != bpe_rank_.end()) {
          heap.push({it->second.value, it->second.id, p, nodes[p].token_id, nodes[i].token_id});
        }
      }
      // Right pair: (i, after_j) — if after_j exists
      if (after_j != -1) {
        auto it = bpe_rank_.find(GetRankKey(nodes[i].token_id, nodes[after_j].token_id));
        if (it != bpe_rank_.end()) {
          heap.push({it->second.value, it->second.id, i, nodes[i].token_id, nodes[after_j].token_id});
        }
      }
    }

    // Write back results
    vals.clear();
    for (int32_t i = 0; i != -1; i = nodes[i].next) {
      vals.emplace_back(nodes[i].token_id, nodes[i].byte_len);
    }
  }

  // =========================================================================
  // Speculative O(N) BPE: greedy longest-match via trie + boundary verification.
  // Returns true if speculation succeeded (vals updated), false if fallback needed.
  // =========================================================================
  bool PerformBPE_Speculative(std::vector<std::pair<uint32_t, uint32_t>>& vals) const {
    const size_t n = vals.size();

    // Greedy longest-match through the trie
    std::vector<std::pair<uint32_t, uint32_t>> output;
    output.reserve(n / 2);  // typical compression ~2:1

    size_t pos = 0;
    while (pos < n) {
      // Walk trie from current position, tracking longest match
      uint32_t cur_node = 0;  // root
      uint32_t best_token_id = vals[pos].first;  // fallback: single atom
      uint32_t best_byte_len = vals[pos].second;
      size_t best_end = pos + 1;

      size_t walk = pos;
      while (walk < n) {
        auto it = spec_trie_[cur_node].children.find(vals[walk].first);
        if (it == spec_trie_[cur_node].children.end()) break;
        cur_node = it->second;
        walk++;

        // Check if this node represents a valid token
        if (spec_trie_[cur_node].token_id != bpe::kInvalidTokenId) {
          best_token_id = spec_trie_[cur_node].token_id;
          // Sum byte lengths for the matched span
          uint32_t total_bytes = 0;
          for (size_t k = pos; k < walk; k++) {
            total_bytes += vals[k].second;
          }
          best_byte_len = total_bytes;
          best_end = walk;
        }
      }

      output.emplace_back(best_token_id, best_byte_len);
      pos = best_end;
    }

    // If no compression happened, speculation trivially succeeds
    if (output.size() == n) {
      // No merges possible — same as BPE would produce
      return true;
    }

    // Boundary verification: check that no cross-boundary merge would have
    // fired before the internal merges that form each token.
    for (size_t i = 0; i + 1 < output.size(); i++) {
      uint32_t ti_id = output[i].first;
      uint32_t ti1_id = output[i + 1].first;

      // Check 1: If merge(ti, ti+1) exists, BPE would merge them further → FAIL
      if (bpe_rank_.find(GetRankKey(ti_id, ti1_id)) != bpe_rank_.end()) {
        return false;
      }

      // Check 2: Cross-boundary atom-level merge check.
      // If merge(last_atom_of_ti, first_atom_of_ti+1) has rank < min_formation_rank
      // of either token, BPE would have chosen a different split → FAIL
      if (ti_id < spec_token_meta_.size() && ti1_id < spec_token_meta_.size()) {
        uint32_t last_atom = spec_token_meta_[ti_id].last_atom;
        uint32_t first_atom = spec_token_meta_[ti1_id].first_atom;
        if (last_atom != bpe::kInvalidTokenId && first_atom != bpe::kInvalidTokenId) {
          auto cross_it = bpe_rank_.find(GetRankKey(last_atom, first_atom));
          if (cross_it != bpe_rank_.end()) {
            uint32_t cross_rank = cross_it->second.value;
            uint32_t min_rank = (std::min)(spec_token_meta_[ti_id].min_formation_rank,
                                           spec_token_meta_[ti1_id].min_formation_rank);
            if (cross_rank < min_rank) {
              return false;
            }
          }
        }
      }

      // Check 3: Multi-character cross-boundary merges.
      // For each suffix of ti's char decomposition that is a known token,
      // and each prefix of ti+1's char decomposition that is a known token,
      // check if their merge would fire before internal merges.
      if (ti_id < spec_token_meta_.size() && ti1_id < spec_token_meta_.size()) {
        const auto& ti_chars = spec_token_meta_[ti_id].char_ids;
        const auto& ti1_chars = spec_token_meta_[ti1_id].char_ids;
        uint32_t min_rank = (std::min)(spec_token_meta_[ti_id].min_formation_rank,
                                       spec_token_meta_[ti1_id].min_formation_rank);

        if (!ti_chars.empty() && !ti1_chars.empty() && min_rank > 0) {
          // Check suffixes of ti (skip full token, already checked above)
          for (size_t s = 1; s < ti_chars.size(); s++) {
            // Look up suffix [s..end] as a token via trie
            uint32_t suffix_id = LookupCharSequence(ti_chars.data() + s, ti_chars.size() - s);
            if (suffix_id == bpe::kInvalidTokenId) continue;

            // Check prefixes of ti+1 (skip full token)
            for (size_t p = 1; p <= ti1_chars.size(); p++) {
              uint32_t prefix_id = LookupCharSequence(ti1_chars.data(), p);
              if (prefix_id == bpe::kInvalidTokenId) continue;

              auto merge_it = bpe_rank_.find(GetRankKey(suffix_id, prefix_id));
              if (merge_it != bpe_rank_.end() && merge_it->second.value < min_rank) {
                return false;
              }
            }
          }
        }
      }
    }

    // All checks passed — speculation succeeded
    vals = std::move(output);
    return true;
  }

  // Look up a sequence of atom token IDs in the trie, return the token ID if found.
  uint32_t LookupCharSequence(const uint32_t* ids, size_t len) const {
    uint32_t cur = 0;
    for (size_t i = 0; i < len; i++) {
      auto it = spec_trie_[cur].children.find(ids[i]);
      if (it == spec_trie_[cur].children.end()) return bpe::kInvalidTokenId;
      cur = it->second;
    }
    return spec_trie_[cur].token_id;
  }

 public:
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

  bool IsNoOpPretokenizer() const { return no_op_pretokenizer_; }

  struct SequencePreTokenizerStep {
    std::string regex;
  };

  bool HasSequencePreTokenizer() const { return !sequence_steps_.empty(); }

  const std::vector<SequencePreTokenizerStep>& GetSequenceSteps() const { return sequence_steps_; }

  std::string GetPreTokenizerRegex(const std::string& model_name, bool spm_model = false) const {
    if (!pre_tokenizer_regex_.empty()) {
      return pre_tokenizer_regex_;
    }

    if (model_name == "Llama" || spm_model) {
      return bpe::PreTokenizerWithRegEx::LLAMA_REGEX_PATTERN;
    }

    // by default, use the GPT2 pretokenizer regex
    return bpe::PreTokenizerWithRegEx::GPT2_REGEX_PATTERN;
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

  // =========================================================================
  // Speculative BPE trie structures
  // =========================================================================
  struct SpecTrieNode {
    std::unordered_map<uint32_t, uint32_t> children;  // atom_token_id → child node index
    uint32_t token_id = bpe::kInvalidTokenId;
  };

  struct SpecTokenMeta {
    std::vector<uint32_t> char_ids;  // character (atom) decomposition
    uint32_t min_formation_rank = (std::numeric_limits<uint32_t>::max)();
    uint32_t first_atom = bpe::kInvalidTokenId;
    uint32_t last_atom = bpe::kInvalidTokenId;
  };

  std::vector<SpecTrieNode> spec_trie_;       // node 0 = root
  std::vector<SpecTokenMeta> spec_token_meta_;
  bool spec_trie_valid_ = false;

  void BuildSpeculativeTrie() {
    spec_trie_valid_ = false;

    // 1. Collect all merges in a sortable list
    struct MergeEntry {
      uint32_t left_id;
      uint32_t right_id;
      uint32_t result_id;
      uint32_t rank;
    };
    std::vector<MergeEntry> merges;
    merges.reserve(bpe_rank_.size());

    for (const auto& [key, node] : bpe_rank_) {
      uint32_t left_id = static_cast<uint32_t>(key & 0xFFFFFFFF);
      uint32_t right_id = static_cast<uint32_t>(key >> 32);
      merges.push_back({left_id, right_id, node.id, node.value});
    }

    // Sort merges by rank (ascending) so we process them in BPE order
    std::sort(merges.begin(), merges.end(),
              [](const MergeEntry& a, const MergeEntry& b) { return a.rank < b.rank; });

    // 2. Initialize token metadata
    // ANY token that appears as an operand in a merge is a potential "atom" — it can
    // appear directly in the input to PerformBPE (e.g., from ByteEncode/SpmTokenize).
    // Even if it's also produced by another merge, we treat it as char_ids={self}
    // because the input has it at that level.
    std::unordered_set<uint32_t> appears_in_merge;
    for (const auto& m : merges) {
      appears_in_merge.insert(m.left_id);
      appears_in_merge.insert(m.right_id);
    }

    size_t max_token_id = id2token_map_.size();
    spec_token_meta_.resize(max_token_id);

    // All operand tokens get char_ids = {self} — they're the "atoms" from PerformBPE's perspective.
    for (uint32_t tid : appears_in_merge) {
      if (tid >= max_token_id) continue;
      spec_token_meta_[tid].char_ids = {tid};
      spec_token_meta_[tid].first_atom = tid;
      spec_token_meta_[tid].last_atom = tid;
    }

    // 3. Process merges in rank order to build char decompositions for merged tokens.
    // Only set char_ids for tokens that DON'T already have them (i.e., non-operand tokens).
    for (const auto& m : merges) {
      if (m.result_id >= max_token_id) continue;
      if (m.left_id >= max_token_id || m.right_id >= max_token_id) continue;

      const auto& left_meta = spec_token_meta_[m.left_id];
      const auto& right_meta = spec_token_meta_[m.right_id];

      // Skip if we don't have char decompositions for both operands
      if (left_meta.char_ids.empty() || right_meta.char_ids.empty()) continue;

      auto& result_meta = spec_token_meta_[m.result_id];
      // Only set if not already set (operands keep {self}; first merge wins for non-operands)
      if (result_meta.char_ids.empty()) {
        result_meta.char_ids.reserve(left_meta.char_ids.size() + right_meta.char_ids.size());
        result_meta.char_ids.insert(result_meta.char_ids.end(),
                                    left_meta.char_ids.begin(), left_meta.char_ids.end());
        result_meta.char_ids.insert(result_meta.char_ids.end(),
                                    right_meta.char_ids.begin(), right_meta.char_ids.end());
        result_meta.first_atom = left_meta.first_atom;
        result_meta.last_atom = right_meta.last_atom;
        result_meta.min_formation_rank = (std::min)(m.rank,
            (std::min)(left_meta.min_formation_rank, right_meta.min_formation_rank));
      }
    }

    // 4. BPE-consistency check + trie construction.
    // Only insert a token into the trie if running BPE on its char_ids produces
    // exactly that token. This prevents over-merging (greedy picking tokens that
    // BPE's merge ordering wouldn't actually produce).
    spec_trie_.clear();
    spec_trie_.emplace_back();  // root node (index 0)

    size_t tokens_inserted = 0;
    for (uint32_t tid = 0; tid < static_cast<uint32_t>(max_token_id); tid++) {
      const auto& meta = spec_token_meta_[tid];
      if (meta.char_ids.empty()) continue;
      if (meta.char_ids.size() < 2) continue;  // don't insert atoms (single char)

      // BPE-consistency check: run BPE on this token's char_ids and verify
      // it produces exactly this token as a single output.
      std::vector<std::pair<uint32_t, uint32_t>> test_vals;
      test_vals.reserve(meta.char_ids.size());
      for (uint32_t cid : meta.char_ids) {
        test_vals.emplace_back(cid, 1);  // byte_len=1 placeholder
      }
      PerformBPE_FlatScan(test_vals);
      if (test_vals.size() != 1 || test_vals[0].first != tid) {
        continue;  // BPE-inconsistent: skip this token
      }

      // Insert char_ids path into trie
      uint32_t cur = 0;
      for (uint32_t cid : meta.char_ids) {
        auto it = spec_trie_[cur].children.find(cid);
        if (it == spec_trie_[cur].children.end()) {
          uint32_t new_idx = static_cast<uint32_t>(spec_trie_.size());
          spec_trie_[cur].children[cid] = new_idx;
          spec_trie_.emplace_back();
          cur = new_idx;
        } else {
          cur = it->second;
        }
      }
      spec_trie_[cur].token_id = tid;
      tokens_inserted++;
    }

    // Also insert all operand tokens (atoms) into the trie as single-step paths.
    for (uint32_t tid : appears_in_merge) {
      if (tid >= max_token_id) continue;
      auto it = spec_trie_[0].children.find(tid);
      if (it == spec_trie_[0].children.end()) {
        uint32_t new_idx = static_cast<uint32_t>(spec_trie_.size());
        spec_trie_[0].children[tid] = new_idx;
        spec_trie_.emplace_back();
        spec_trie_[new_idx].token_id = tid;
      } else {
        // Node already exists (from a longer path), ensure it's marked as valid
        if (spec_trie_[it->second].token_id == bpe::kInvalidTokenId) {
          spec_trie_[it->second].token_id = tid;
        }
      }
    }

    spec_trie_valid_ = (tokens_inserted > 0);
  }

 private:
  std::string end_of_word_suffix_;
  std::unordered_map<uint64_t, BpeNode> bpe_rank_;

  std::unordered_map<std::string, uint32_t> vocab_map_;
  std::vector<std::string> id2token_map_;

  uint32_t unk_id_ = (std::numeric_limits<uint32_t>::max)();
  bpe::SpecialTokenMap special_tokens_;
  TrieTree<char32_t> added_tokens_;
  std::string pre_tokenizer_regex_;
  bool no_op_pretokenizer_ = false;
  std::vector<SequencePreTokenizerStep> sequence_steps_;

  std::set<std::string_view> pre_tokenizer_types_;

  static std::string NormalizeJsonRegexEscapes(std::string regex) {
    std::string normalized;
    normalized.reserve(regex.size());
    for (char ch : regex) {
      switch (ch) {
        case '\r': normalized += "\\r"; break;
        case '\n': normalized += "\\n"; break;
        case '\t': normalized += "\\t"; break;
        case '\f': normalized += "\\f"; break;
        case '\v': normalized += "\\v"; break;
        default:   normalized += ch;    break;
      }
    }
    return normalized;
  }
};

}  // namespace ort_extensions
