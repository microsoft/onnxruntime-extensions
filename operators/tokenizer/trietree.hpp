// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "ocos.h"
#include "narrow.h"

#include <vector>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <charconv>
#include <optional>

class TrieTree {
 public:
  static constexpr int kMaxTokenLength_ = 128;

  TrieTree(unsigned char ch = 0) : ch_(ch), to_(256) {}

  void Add(const std::string& key, int idx = 0,
           std::optional<int> value = std::optional<int>()) {
    if (idx == key.length()) {
      if (!value) {
        value = key[0];
      }
      value_ = value;
      return;
    }

    unsigned char ch = static_cast<unsigned char>(key[idx]);
    if (to_[ch] == nullptr) {
      to_[ch] = std::make_unique<TrieTree>(ch);
    }
    to_[ch]->Add(key, idx + 1, value);
  }

  int FindLongest(const std::string& key, size_t& idx) {
    const TrieTree* u = this;
    unsigned char ch = key[idx];

    int tok_id = 0;
    size_t idx_end = idx;
    while (u->to_[ch]) {
      u = u->to_[ch].get();
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

 private:
  std::vector<std::unique_ptr<TrieTree>> to_;
  std::optional<int> value_;
  unsigned char ch_;
};
