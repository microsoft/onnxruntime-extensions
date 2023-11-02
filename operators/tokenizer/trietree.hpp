// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "ocos.h"
#include "narrow.h"

#include <vector>
#include <set>
#include <map>
#include <string>
#include <optional>

template <typename CharT, typename ValueT = int>
class TrieTree {
 public:
  static constexpr int kMaxTokenLength_ = 128;

  TrieTree(CharT ch = 0, ValueT invalid_id = -1) : ch_(ch), invalid_id_(invalid_id) {}

  void Add(const std::basic_string<CharT>& key, int idx = 0,
           std::optional<ValueT> value = std::optional<ValueT>()) noexcept {
    if (idx == key.length()) {
      if (!value) {
        value = key[0];
      }
      value_ = value;
      return;
    }

    auto ch = key[idx];
    if (to_.count(ch) == 0) {
      to_[ch] = std::make_unique<TrieTree>(ch);
    }
    to_[ch]->Add(key, idx + 1, value);
  }

  ValueT FindLongest(const std::basic_string<CharT>& key, size_t& idx) const noexcept {
    const TrieTree* u = this;
    CharT ch = key[idx];

    ValueT tok_id = invalid_id_;
    size_t idx_end = idx;
    while (u->to_.count(ch)) {
      u = u->to_.at(ch).get();
      idx += 1;
      if (u->value_) {
        tok_id = *u->value_;
        idx_end = idx;
      }
      if (idx == key.length()) {
        break;
      }
      ch = key[idx];
    }

    idx = idx_end;
    return tok_id;
  }

  int Split(const std::basic_string<CharT>& input, std::vector<std::pair<std::basic_string_view<CharT>, ValueT>>& tokens) const noexcept {
    size_t seg_idx = 0;
    size_t tok_idx = 0;

    while (tok_idx < input.length()) {
      // variable u is the tree root.
      const TrieTree* u = this;
      auto ch = input[tok_idx];
      size_t tok_len = 0;
      size_t idx_end = 0;
      ValueT tok_id = invalid_id_;

      // try to match a longest token
      while (u->to_.count(ch)) {
        tok_len += 1;
        u = u->to_.at(ch).get();
        if (u->value_) {
          tok_id = *u->value_;
          idx_end = tok_idx;
        }

        tok_idx += 1;
        if (tok_idx == input.length()) {
          break;
        }
        ch = input[tok_idx];
      }
      if (tok_idx == seg_idx || tok_len == 0) {
        tok_idx += 1;
        if (tok_idx < input.length()) {
          continue;
        }
      }

      auto token_begin_idx = tok_idx - tok_len;
      if (token_begin_idx > seg_idx) {
        tokens.emplace_back(std::basic_string_view<CharT>(input.data() + seg_idx, token_begin_idx - seg_idx), invalid_id_);
      }
      if (tok_len > 0) {
        tokens.emplace_back(std::basic_string_view<CharT>(input.data() + token_begin_idx, idx_end - seg_idx), tok_id);
        tok_idx = idx_end;
      }

      // reset state for next match
      seg_idx = tok_idx;
    }

    return 0;
  }

 private:
  std::map<CharT, std::unique_ptr<TrieTree>> to_;
  std::optional<ValueT> value_;
  const CharT ch_;
  const ValueT invalid_id_;
};
