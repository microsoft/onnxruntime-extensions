// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <regex>
#include <cassert>
#include <algorithm>

#include "ustring.h"
#include "unicode.h"
#include "ext_status.h"

namespace ort_extensions {
namespace bpe {

using TokenPairs = std::vector<std::pair<std::u32string_view, int>>;

constexpr int kInvalidTokenId = -1;

class SpecialTokenMap {
 public:
  OrtxStatus Add(ustring p_str, int p_id) {
    if (p_str.empty()) {
      return {kOrtxErrorInvalidArgument, "Empty special token."};
    }
    auto it = token_map_.find(p_str);
    if (it != token_map_.end()) {
      assert(it->second == p_id && "Duplicate special tokens.");
    } else {
      token_map_[p_str] = p_id;
      token_list_.push_back(SpecialTokenInfo(std::move(p_str), p_id));
    }

    return {};
  }

  TokenPairs SplitBySpecialTokens(const std::u32string_view& input) const {
    TokenPairs res;
    res.emplace_back(input, kInvalidTokenId);
    for (const auto& st : token_list_) {
      TokenPairs new_split_res;
      for (auto& str : res) {
        if (str.second != kInvalidTokenId) {
          new_split_res.emplace_back(str);
          continue;
        }

        auto it = str.first.begin();
        size_t search_pos = 0;
        while (it != str.first.end()) {
// works fine for all clang-based platform: Mac OS, Android, WebAssembly
#if defined(__clang__)
          auto search_it = std::search(it, str.first.end(), st.str.begin(), st.str.end());
#else
          auto search_it = std::search(it, str.first.end(), std::boyer_moore_searcher(st.str.begin(), st.str.end()));
#endif
          if (search_it == str.first.end()) {
            new_split_res.emplace_back(std::u32string_view(str.first.data() + search_pos, str.first.size() - search_pos),
                                       kInvalidTokenId);
            break;
          }

          auto prefixLen = search_it - it;
          if (prefixLen != 0) {
            new_split_res.emplace_back(std::u32string_view(str.first.data() + search_pos, prefixLen), kInvalidTokenId);
            search_pos += prefixLen;
          }

          new_split_res.emplace_back(std::u32string_view(str.first.data() + search_pos, st.str.size()), st.id);
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

    SpecialTokenInfo(ustring p_str, int p_id) : str(std::move(p_str)), id(p_id) {}
  };

  std::list<SpecialTokenInfo> token_list_;
  std::unordered_map<ustring, int> token_map_;
};

class PreTokenizerWithRegEx {
 public:
  static constexpr const char GPT2_REGEX_PATTERN[] =
      R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
  static constexpr const char LLAMA_REGEX_PATTERN[] =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

  PreTokenizerWithRegEx() = default;

  // "'s|'t|'re|'ve|'m|'ll|'d"
  std::u32string_view Match_GPT2_Pattern_1() {
    if ((m_text[0] == U'\'') && (m_text.size() > 1)) {
      if ((m_text[1] == U's') || (m_text[1] == U't') || (m_text[1] == U'm') || (m_text[1] == U'd')) {
        std::u32string_view res = m_text.substr(0, 2);
        m_text = m_text.substr(2);
        return res;
      }

      if (m_text.size() > 2) {
        if (((m_text[1] == U'r') && (m_text[2] == U'e')) || ((m_text[1] == U'v') && (m_text[2] == U'e')) ||
            ((m_text[1] == U'l') && (m_text[2] == U'l'))) {
          std::u32string_view res = m_text.substr(0, 3);
          m_text = m_text.substr(3);
          return res;
        }
      }
    }

    return {};
  }

  // " ?\p{L}+| ?\p{N}+"
  std::u32string_view Match_GPT2_Pattern_2() {
    // ?\p{L}+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && IsL(m_text[1])) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if (!IsL(m_text[i])) break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (IsL(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsL(m_text[i])) break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // ?\p{N}+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && IsN(m_text[1])) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if (!IsN(m_text[i])) break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (IsN(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsN(m_text[i])) break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    return {};
  }

  // " ?[^\s\p{L}\p{N}]+"
  std::u32string_view Match_GPT2_Pattern_3() {
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (NotLNZ(m_text[1]))) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i])) break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (NotLNZ(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i])) break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    return {};
  }

  // "\s+(?!\S)|\s+)"
  std::u32string_view Match_GPT2_Pattern_4() {
    if ((m_text.size() >= 1) && (IsZ(m_text[0]))) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsZ(m_text[i])) break;
      }
      if ((i > 1) && (i != m_text.size())) {  //\s+(?!\S)
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

    return {};
  }

  // [\p{L}]+|[\p{N}]
  std::u32string_view Match_CLIP_Pattern_1() {
    if (IsL(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsL(m_text[i])) break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    if (IsN(m_text[0])) {
      std::u32string_view res = m_text.substr(0, 1);
      m_text = m_text.substr(1);
      return res;
    }

    return {};
  }

  // [^\s\p{L}\p{N}]+
  std::u32string_view Match_CLIP_Pattern_2() {
    if (NotLNZ(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i])) break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    return {};
  }

  // "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
  // (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
  std::u32string_view Match_LLAMA3_Pattern_1() {
    if ((m_text[0] == U'\'') && (m_text.size() > 1)) {
      if ((m_text[1] == U's') || (m_text[1] == U't') || (m_text[1] == U'm') || (m_text[1] == U'd') ||
          (m_text[1] == U'S') || (m_text[1] == U'T') || (m_text[1] == U'M') || (m_text[1] == U'D')) {
        std::u32string_view res = m_text.substr(0, 2);
        m_text = m_text.substr(2);
        return res;
      }

      if (m_text.size() > 2) {
        if ((((m_text[1] == U'r') || (m_text[1] == U'R')) && ((m_text[2] == U'e') || (m_text[2] == U'E'))) ||
            (((m_text[1] == U'v') || (m_text[1] == U'V')) && ((m_text[2] == U'e') || (m_text[2] == U'E'))) ||
            (((m_text[1] == U'l') || (m_text[1] == U'L')) && ((m_text[2] == U'l') || (m_text[2] == U'L')))) {
          std::u32string_view res = m_text.substr(0, 3);
          m_text = m_text.substr(3);
          return res;
        }
      }
    }

    return {};
  }

  // "[^\r\n\p{L}\p{N}]?\p{L}+"
  std::u32string_view Match_LLAMA3_Pattern_2() {
    if ((!IsRN(m_text[0]) && !IsN(m_text[0])) || IsL(m_text[0])) {
      if (IsL(m_text[0]) || ((m_text.size() > 1) && IsL(m_text[1]))) {
        size_t i = 1;
        for (; i < m_text.size(); ++i) {
          if (!IsL(m_text[i])) break;
        }
        std::u32string_view res = m_text.substr(0, i);
        m_text = m_text.substr(i);
        return res;
      }
    }

    return {};
  }

  // "\p{N}{1,3}"
  std::u32string_view Match_LLAMA3_Pattern_3() {
    if (IsN(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsN(m_text[i]) || (i > 2)) break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    return {};
  }

  // " ?[^\s\p{L}\p{N}]+[\r\n]*"
  std::u32string_view Match_LLAMA3_Pattern_4() {
    auto pos = 0;
    if (m_text[0] == U' ') pos = 1;
    if (pos < m_text.size() && NotLNZ(m_text[pos])) {
      size_t i = pos + 1;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i])) break;
      }
      if (i < m_text.size() && IsRN(m_text[i])) {
        for (; i < m_text.size(); ++i) {
          if (!IsRN(m_text[i])) break;
        }
      }
      std::u32string_view res = m_text.substr(pos, i - pos);
      m_text = m_text.substr(i);
      return res;
    }

    return {};
  }

  // "\s*[\r\n]+"
  std::u32string_view Match_LLAMA3_Pattern_5() {
    if (IsZ(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsZ(m_text[i])) break;
      }
      if (i < m_text.size() && IsRN(m_text[i])) {
        for (; i < m_text.size(); ++i) {
          if (!IsRN(m_text[i])) break;
        }
      }
      if (i > 1 && IsRN(m_text[i - 1])) {
        std::u32string_view res = m_text.substr(0, i);
        m_text = m_text.substr(i);
        return res;
      }
    }
    if (IsRN(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsRN(m_text[i])) break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    return {};
  }

  void CategoryMatch(size_t& index, std::set<ufal::unilib::unicode::category_t>& categories){
    while (index < m_text.size() && categories.find(ufal::unilib::unicode::category(m_text[index])) != categories.end()){
      index++;
    }
  }

  // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
  std::u32string_view Match_PHI4_Pattern_1() {
    size_t i = 0;

    // [^\r\n\p{L}\p{N}]?
    if (!IsRN(m_text[i]) && !IsN(m_text[i]) && !IsL(m_text[i])) {
      i++;
    }

    // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*
    std::set<ufal::unilib::unicode::category_t> categories1 = {ufal::unilib::unicode::Lu,
                                                               ufal::unilib::unicode::Lt,
                                                               ufal::unilib::unicode::Lm,
                                                               ufal::unilib::unicode::Lo,
                                                               ufal::unilib::unicode::M};
    CategoryMatch(i, categories1);

    // [\p{Ll}\p{Lm}\p{Lo}\p{M}]+
    size_t j = i;
    std::set<ufal::unilib::unicode::category_t> categories2 = {ufal::unilib::unicode::Ll,
                                                                 ufal::unilib::unicode::Lm,
                                                                 ufal::unilib::unicode::Lo,
                                                                 ufal::unilib::unicode::M};
    CategoryMatch(i, categories2);
    if (i == j){
      // No case match, return as this is a '+' category case (one or more occurrences must be found)
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)?
    if ((m_text[i] == U'\'') && ((i + 1) < m_text.size())) {
      if ((m_text[i + 1] == U's') || (m_text[i + 1] == U't') || (m_text[i + 1] == U'm') || (m_text[i + 1] == U'd') ||
          (m_text[i + 1] == U'S') || (m_text[i + 1] == U'T') || (m_text[i + 1] == U'M') || (m_text[i + 1] == U'D')) {
        i += 2;
      } else if ((i + 2) < m_text.size()) {
        if ((((m_text[i + 1] == U'r') || (m_text[i + 1] == U'R')) && ((m_text[i + 2] == U'e') || (m_text[i + 2] == U'E'))) ||
            (((m_text[i + 1] == U'v') || (m_text[i + 1] == U'V')) && ((m_text[i + 2] == U'e') || (m_text[i + 2] == U'E'))) ||
            (((m_text[i + 1] == U'l') || (m_text[i + 1] == U'L')) && ((m_text[i + 2] == U'l') || (m_text[i + 2] == U'L')))) {
          i += 3;
        }
      }
    }

    std::u32string_view res = m_text.substr(0, i);
    m_text = m_text.substr(i);
    return res;
  }

  // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
  std::u32string_view Match_PHI4_Pattern_2() {
    size_t i = 0;

    // [^\r\n\p{L}\p{N}]?
    if (!IsRN(m_text[i]) && !IsN(m_text[i]) && !IsL(m_text[i])) {
      i++;
    }

    // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+
    size_t j = i;
    std::set<ufal::unilib::unicode::category_t> categories1 = {ufal::unilib::unicode::Lu,
                                                                 ufal::unilib::unicode::Lt,
                                                                 ufal::unilib::unicode::Lm,
                                                                 ufal::unilib::unicode::Lo,
                                                                 ufal::unilib::unicode::M};
    CategoryMatch(i, categories1);
    if (i == j){
      // No case match, return as this is a '+' category case (one or more occurrences must be found)
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // [\p{Ll}\p{Lm}\p{Lo}\p{M}]*
    std::set<ufal::unilib::unicode::category_t> categories2 = {ufal::unilib::unicode::Ll,
                                                                 ufal::unilib::unicode::Lm,
                                                                 ufal::unilib::unicode::Lo,
                                                                 ufal::unilib::unicode::M};
    CategoryMatch(i, categories2);

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)?
    if ((m_text[i] == U'\'') && ((i + 1) < m_text.size())) {
      if ((m_text[i + 1] == U's') || (m_text[i + 1] == U't') || (m_text[i + 1] == U'm') || (m_text[i + 1] == U'd') ||
          (m_text[i + 1] == U'S') || (m_text[i + 1] == U'T') || (m_text[i + 1] == U'M') || (m_text[i + 1] == U'D')) {
        i += 2;
      } else if ((i + 2) < m_text.size()) {
        if ((((m_text[i + 1] == U'r') || (m_text[i + 1] == U'R')) && ((m_text[i + 2] == U'e') || (m_text[i + 2] == U'E'))) ||
            (((m_text[i + 1] == U'v') || (m_text[i + 1] == U'V')) && ((m_text[i + 2] == U'e') || (m_text[i + 2] == U'E'))) ||
            (((m_text[i + 1] == U'l') || (m_text[i + 1] == U'L')) && ((m_text[i + 2] == U'l') || (m_text[i + 2] == U'L')))) {
          i += 3;
        }
      }
    }

    std::u32string_view res = m_text.substr(0, i);
    m_text = m_text.substr(i);
    return res;
  }

  // "(\p{N})"
  std::u32string_view Match_General_Pattern_1() {
    if (IsN(m_text[0])) {
      std::u32string_view res = m_text.substr(0, 1);
      m_text = m_text.substr(1);
      return res;
    }

    return {};
  }

  using RegexMatchFunc = std::u32string_view (PreTokenizerWithRegEx::*)();
  OrtxStatus Compile(const std::string& regex) {
    // NOTES: to avoid the short pattern shadowing the longer one, the longer pattern should be placed first
    auto patterns = std::vector<std::tuple<std::string_view, RegexMatchFunc>>{
        {R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]))",
         &PreTokenizerWithRegEx::Match_LLAMA3_Pattern_1},
        {R"((?i:'s|'t|'re|'ve|'m|'ll|'d))", &PreTokenizerWithRegEx::Match_LLAMA3_Pattern_1},
        {R"('s|'t|'re|'ve|'m|'ll|'d)", &PreTokenizerWithRegEx::Match_GPT2_Pattern_1},
        {R"([^\r\n\p{L}\p{N}]?\p{L}+)", &PreTokenizerWithRegEx::Match_LLAMA3_Pattern_2},
        {R"(\p{N}{1,3})", &PreTokenizerWithRegEx::Match_LLAMA3_Pattern_3},
        {R"( ?[^\s\p{L}\p{N}]+[\r\n]*)", &PreTokenizerWithRegEx::Match_LLAMA3_Pattern_4},
        {R"(\s*[\r\n]+)", &PreTokenizerWithRegEx::Match_LLAMA3_Pattern_5},
        {R"( ?\p{L}+| ?\p{N}+)", &PreTokenizerWithRegEx::Match_GPT2_Pattern_2},
        {R"( ?[^\s\p{L}\p{N}]+)", &PreTokenizerWithRegEx::Match_GPT2_Pattern_3},
        {R"(\s+(?!\S)|\s+)", &PreTokenizerWithRegEx::Match_GPT2_Pattern_4},
        {R"([\p{L}]+|[\p{N}])", &PreTokenizerWithRegEx::Match_CLIP_Pattern_1},
        {R"([^\s\p{L}\p{N}]+)", &PreTokenizerWithRegEx::Match_CLIP_Pattern_2},
        {R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?)", &PreTokenizerWithRegEx::Match_PHI4_Pattern_1},
        {R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?)", &PreTokenizerWithRegEx::Match_PHI4_Pattern_2},
        {R"(?[^\s\p{L}\p{N}]+[\r\n/]*)", &PreTokenizerWithRegEx::Match_LLAMA3_Pattern_4},
        {R"(\p{N})", &PreTokenizerWithRegEx::Match_General_Pattern_1},
    };

    std::string regex_compound = regex;
    for (const auto& [pattern, func] : patterns) {
      auto pos = regex_compound.find(pattern);
      if (pos != std::string::npos) {
        if (pos > 0) {
          if (regex_compound[pos - 1] != '|') {
            continue;
          }
        }
        if (pos + pattern.size() < regex_compound.size()) {
          if (regex_compound[pos + pattern.size()] != '|') {
            continue;
          }
        }

        activated_matchers_.push_back(func);
        std::string regex_prefix;
        auto pattern_size = pattern.size();
        if (pos > 0) {  // remove the '|' at the end of the prefix
          regex_prefix = regex_compound.substr(0, pos);
          if (regex_prefix.back() == '|') {
            regex_prefix.pop_back();
          }
        } else {
          if (pattern_size < regex_compound.size()) {
            assert(regex_compound[pattern_size] == '|');
            pattern_size++; // let the pattern include the '|'
          }
        }
        regex_compound = regex_prefix + regex_compound.substr(pos + pattern_size);
      }
    }

    if (regex_compound.size() > 0) {
      try {
        fallback_patterns_ = std::make_unique<std::regex>(regex_compound);
      } catch (const std::regex_error& e) {
        return {kOrtxErrorInvalidArgument, "Invalid regex: " + regex_compound + "\n" + e.what()};
      }
    }

    return {};
  }

  std::u32string_view TryMatch() {
    std::u32string_view res;
    for (auto& matcher : activated_matchers_) {
      res = (this->*matcher)();
      if (!res.empty()) {
        break;
      }
    }

    if (fallback_patterns_) {
      if (res.empty()) {
        res = MatchWithSTLRegEx();
      } else {
        // update the m_utf8_text for the next iteration
        std::string utf8_res = (std::string)ustring(res);
        auto pos = m_utf8_text.find(utf8_res);
        assert(pos != std::string::npos);
        m_utf8_text = m_utf8_text.substr(pos + utf8_res.size());
      }
    }

    return res;
  }

  std::u32string_view MatchWithSTLRegEx() {
    std::smatch match;
    auto& text = m_utf8_text;
    if (std::regex_search(text, match, *fallback_patterns_)) {
      ustring res(match[0]);
      auto pos = m_text.find(res);
      m_utf8_text = m_utf8_text.substr(pos + match[0].str().size());
      auto res_view = m_text.substr(pos, res.size());
      m_text = m_text.substr(pos + res.size());
      return res_view;
    }

    return {};
  }

  void Set(std::u32string_view val) {
    m_text = val;
    m_last_char = U'\0';
    if (fallback_patterns_) {
      m_utf8_text = (std::string)(ustring(val));
    }
  }

  // always return a token until the end of the string
  std::u32string_view GetNextToken() {
    while (!m_text.empty()) {
      auto res = TryMatch();
      if (res.empty()) {
        m_last_char = m_text[0];
        m_text = m_text.substr(1);
        continue;
      }

      m_last_char = res.back();
      return res;
    }

    return {};
  }

 public:
  static bool IsRN(char32_t ch) { return ch == U'\r' || ch == U'\n'; }

  static bool IsL(char32_t ch) {
    auto category = ufal::unilib::unicode::category(ch);
    return (category & ufal::unilib::unicode::L) != 0;
  }

  static bool IsN(char32_t ch) {
    auto category = ufal::unilib::unicode::category(ch);
    return (category & ufal::unilib::unicode::N) != 0;
  }

  static bool IsZ(char32_t ch) {
    if (ch == U'\r' || ch == U'\n' || ch == U'\t' || ch == U'\f' || ch == U'\v') return true;
    auto category = ufal::unilib::unicode::category(ch);
    return (category & ufal::unilib::unicode::Z) != 0;
  }

  static bool NotLNZ(char32_t ch) {
    // \r\n\t\f\v
    if (ch == U'\r' || ch == U'\n' || ch == U'\t' || ch == U'\f' || ch == U'\v') return false;
    auto category = ufal::unilib::unicode::category(ch);
    if (category & ufal::unilib::unicode::L) return false;
    if (category & ufal::unilib::unicode::N) return false;
    if (category & ufal::unilib::unicode::Z) return false;
    return true;
  }

 private:
  std::u32string_view m_text;
  char32_t m_last_char = 0;

  std::vector<RegexMatchFunc> activated_matchers_;
  std::unique_ptr<std::regex> fallback_patterns_;
  std::string m_utf8_text;
};

}  // namespace bpe
}  // namespace ort_extensions
