// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "narrow.h"

#include <algorithm>
#include "ustring.h"

#include "unicode.h"

class SpecialTokenMap {
 public:
  void Add(ustring p_str, int p_id) {
    auto it = token_map_.find(p_str);
    if (it != token_map_.end()) {
      if (it->second != p_id) {
        ORTX_CXX_API_THROW("Duplicate special tokens.", ORT_INVALID_ARGUMENT);
      }
    } else {
      token_map_[p_str] = p_id;
      token_list_.push_back(SpecialTokenInfo(std::move(p_str), p_id));
    }
  }

  std::vector<std::pair<ustring, int>> SplitBySpecialTokens(ustring input) const {
    std::vector<std::pair<ustring, int>> res;
    res.emplace_back(std::move(input), -1);
    for (const auto& st : token_list_) {
      std::vector<std::pair<ustring, int>> new_split_res;
      for (auto& str : res) {
        if (str.second != -1) {
          new_split_res.push_back(std::move(str));
          continue;
        }
        auto it = str.first.begin();
        size_t search_pos = 0;
        while (it != str.first.end()) {
// works fine for all clang-based platform: Mac OS, Android, WebAssembly
#if defined(__clang__)
          auto search_it = std::search(it, str.first.end(), st.str.begin(), st.str.end());
#else
          auto search_it = std::search(it, str.first.end(),
                                       std::boyer_moore_searcher(st.str.begin(), st.str.end()));
#endif
          if (search_it == str.first.end()) {
            new_split_res.emplace_back(str.first.substr(search_pos), -1);
            break;
          }
          auto prefixLen = search_it - it;
          if (prefixLen != 0) {
            new_split_res.emplace_back(str.first.substr(search_pos, prefixLen), -1);
            search_pos += prefixLen;
          }
          new_split_res.emplace_back(str.first.substr(search_pos, st.str.size()), st.id);
          it = search_it + st.str.size();
          search_pos += st.str.size();
        }
      }
      std::swap(new_split_res, res);
    }
    return res;
  }

 private:
  struct SpecialTokenInfo {
    ustring str;
    int id;

    SpecialTokenInfo(ustring p_str, int p_id)
        : str(std::move(p_str)), id(p_id) {
      if (str.empty()) {
        ORTX_CXX_API_THROW("Empty special token.", ORT_INVALID_ARGUMENT);
      }
    }
  };

  std::list<SpecialTokenInfo> token_list_;
  std::unordered_map<ustring, int> token_map_;
};

class TokenWithRegularExp {
 public:
  void Set(std::u32string_view val) {
    m_text = val;
  }

  std::pair<bool, std::u32string_view> GetNextToken() {
    while (!m_text.empty()) {
      auto res = TryMatch();
      if (res.empty()) {
        m_text = m_text.substr(1);
        continue;
      }
      return {true, res};
    }
    return {false, {}};
  }

 private:
  std::u32string_view TryMatch() {

    // python pattern:
    // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

    // 's|'t|'re|'ve|'m|'ll|'d|
    // Note: the sequencial of the following if should not be switched, which follows the python regex's syntax
    if ((m_text[0] == U'\'') && (m_text.size() > 1)) {
      if ((m_text[1] == U's') || (m_text[1] == U't') ||
          (m_text[1] == U'm') || (m_text[1] == U'd')) {
        std::u32string_view res = m_text.substr(0, 2);
        m_text = m_text.substr(2);
        return res;
      }

      if (m_text.size() > 2) {
        if (((m_text[1] == U'r') && (m_text[2] == U'e')) ||
            ((m_text[1] == U'v') && (m_text[2] == U'e')) ||
            ((m_text[1] == U'l') && (m_text[2] == U'l'))) {
          std::u32string_view res = m_text.substr(0, 3);
          m_text = m_text.substr(3);
          return res;
        }
      }
    }

    // ?\p{L}+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (ufal::unilib::unicode::category(m_text[1]) & ufal::unilib::unicode::L)) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::L) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (ufal::unilib::unicode::category(m_text[0]) & ufal::unilib::unicode::L) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::L) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // ?\p{N}+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (ufal::unilib::unicode::category(m_text[1]) & ufal::unilib::unicode::N)) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::N) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (ufal::unilib::unicode::category(m_text[0]) & ufal::unilib::unicode::N) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::N) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // ?[^\s\p{L}\p{N}]+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (NotLNZ(m_text[1]))) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i]))
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (NotLNZ(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i]))
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // \s+(?!\S)|\s+
    if ((m_text.size() >= 1) && (IsZ(m_text[0]))) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsZ(m_text[i])) break;
      }
      if ((i > 1) && (i != m_text.size()))  //\s+(?!\S)
      {
        i--;
        std::u32string_view res = m_text.substr(0, i);
        m_text = m_text.substr(i);
        return res;
      }
      // \s+
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    return std::u32string_view{};
  }

  static bool IsZ(char32_t ch) {
    auto category = ufal::unilib::unicode::category(ch);
    return (category & ufal::unilib::unicode::Z) != 0;
  }

  static bool NotLNZ(char32_t ch) {
    auto category = ufal::unilib::unicode::category(ch);
    if (category & ufal::unilib::unicode::L) return false;
    if (category & ufal::unilib::unicode::N) return false;
    if (category & ufal::unilib::unicode::Z) return false;
    return true;
  }

 private:
  std::u32string_view m_text;
};

class LRUCache {
  // Store keys of cache
  std::list<std::string> dq;

  // Store references of keys in cache for efficiency
  std::unordered_map<std::string, std::list<std::string>::iterator> references;

  // Store input IDs and offset mappings of tokens
  std::unordered_map<std::string, std::list<std::pair<uint32_t, uint32_t>>> input_ids_and_offsets;
  
  // Maximum capacity of cache
  int capacity;

 public:
  // Declare the size
  LRUCache(int n) { capacity = n; }

  // Add tok to the LRU cache
  void add(std::string tok, std::list<std::pair<uint32_t, uint32_t>> output) {
    // token not present in cache
    if (references.find(tok) == references.end()) {
      // cache is full
      if (dq.size() == capacity) {
        // delete least recently used element
        std::string last = dq.back();

        // pop last element
        dq.pop_back();

        // erase last key reference
        references.erase(last);

        // erase output for key that is being removed from cache
        input_ids_and_offsets.erase(last);
      }
    } else {
      // tok present in cache
      dq.erase(references[tok]);

      // add output for tok
      const std::pair out = std::make_pair(tok, output);
      input_ids_and_offsets.insert(out);
    }

    // update keys and references
    dq.push_front(tok);
    references[tok] = dq.begin();
  }

  // Check if token is already tokenized
  bool already_tokenized(std::string tok) {
    bool tokenized = input_ids_and_offsets.find(tok) != input_ids_and_offsets.end();
    if (tokenized) {
      // update keys and references since tok is now recently used
      dq.erase(references[tok]);
      dq.push_front(tok);
      references[tok] = dq.begin();
    }
    return tokenized;
  }

  // Return output for token that is already tokenized
  std::list<std::pair<uint32_t, uint32_t>> get_output(std::string tok) {
    return input_ids_and_offsets[tok];
  }
};
