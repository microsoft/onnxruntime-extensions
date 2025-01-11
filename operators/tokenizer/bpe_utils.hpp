// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <regex>
#include <cassert>
#include <algorithm>

#include "ustring.h"
#include "unicode.h"

#include <regex>

namespace ort_extensions {
namespace bpe {

using TokenPairs = std::vector<std::pair<std::u32string_view, int>>;
using u32string_view = std::u32string_view;

constexpr int kInvalidTokenId = -1;

class SpecialTokenMap {
 public:
  void Add(ustring p_str, int p_id) {
    auto it = token_map_.find(p_str);
    if (it != token_map_.end()) {
      assert(it->second == p_id && "Duplicate special tokens.");
    } else {
      token_map_[p_str] = p_id;
      token_list_.push_back(SpecialTokenInfo(std::move(p_str), p_id));
    }
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
            new_split_res.emplace_back(u32string_view(str.first.data() + search_pos, str.first.size() - search_pos),
                                       kInvalidTokenId);
            break;
          }

          auto prefixLen = search_it - it;
          if (prefixLen != 0) {
            new_split_res.emplace_back(u32string_view(str.first.data() + search_pos, prefixLen), kInvalidTokenId);
            search_pos += prefixLen;
          }

          new_split_res.emplace_back(u32string_view(str.first.data() + search_pos, st.str.size()), st.id);
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

    SpecialTokenInfo(ustring p_str, int p_id) : str(std::move(p_str)), id(p_id) {
      if (str.empty()) {
        ORTX_CXX_API_THROW("Empty special token.", ORT_INVALID_ARGUMENT);
      }
    }
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
    if (!IsRN(m_text[0]) && !IsN(m_text[0])) {
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
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (NotLNZ(m_text[1]))) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i])) break;
      }
      if (i < m_text.size() && IsRN(m_text[i])) {
        for (; i < m_text.size(); ++i) {
          if (!IsRN(m_text[i])) break;
        }
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
      if (i < m_text.size() && IsRN(m_text[i])) {
        for (; i < m_text.size(); ++i) {
          if (!IsRN(m_text[i])) break;
        }
      }
      std::u32string_view res = m_text.substr(0, i);
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
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
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
    // NOTES: to avoid the short pattern shadowing the longer one, we sort the patterns by length
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
  // Although we have RegexMatchGeneral which performs regex matching given any general regex string
  // (not just GPT2, Llama, or C++ standard library compatible regex strings), we write the following
  // two methods for GPT2 and Llama in order to make performance improvements.

  std::u32string_view RegexMatchGPT2() {
    // Case 1: English apostrophe handling (1st-7th Alternative, the endings for her's, can't, you're, etc.)

    // Note: the sequencial of the following if should not be switched, which follows the python regex's syntax

    // Standard Library Search might be too compute intensive here due to conversions to and fro ustring and wstring,
    // so we stick to manual parsing here for efficiency, however (as an example for the usage of RegexMatchSTD),
    // note that this following snippet would also work:

    // std::u32string_view std_regex = RegexMatchSTD(U"'s|'t|'re|'ve|'m|'ll|'d");
    // if (std_regex.size() != 0){
    //   return std_regex;
    // }

    // 's|'t|'re|'ve|'m|'ll|'d
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

    // Case 2: Language Character Class (8th Alternative)
    // ? matches the previous token between zero and one times, as many times as possible, giving back as needed
    // (greedy) \p{L} matches any kind of letter from any language
    // + matches the previous token between one and unlimited times, as many times as possible, giving back as needed
    // (greedy)

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

    // Case 3: Numeric Character Class (9th Alternative)
    // \p{N} matches any kind of numeric character in any script
    // all other symbols as previously described.

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

    // Case 4: Custom Character Combination (10th Alternative)
    // [^...] matches a single character not present in the list
    // \s matches any whitespace character (equivalent to [\r\n\t\f\v])
    // all other symbols as previously described.

    // ?[^\s\p{L}\p{N}]+
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

    // Case 5: Custom Character Combination (11th and 12th Alternative)
    // (?!...) is a Negative Lookahead, it asserts that the regex in ... does not match
    // \S matches any non-whitespace character (equivalent to [^\r\n\t\f\v])
    // all other symbols as previously described.

    // \s+(?!\S)|\s+
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

    return U"";
  }

  std::u32string_view RegexMatchLlama3() {
    // Case 1: English apostrophe handling, case insensitive (1st Alternative, the endings for her's, CAN'T, etc.)
    // (?_: ...) is a Non-capturing Group, which matches the tokens contained with the effective flags following ?
    // i modifier performs a case insensitive match (ignores case of [a-zA-Z])
    // all other symbols as previously described.

    // Note: the sequencial of the following if should not be switched, which follows the python regex's syntax

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
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

    // Case 2: Custom Character Combination (2nd Alternative)
    // \r matches a carriage return (ASCII 13)
    // \n matches a line-feed (newline) character (ASCII 10)
    // all other symbols as previously described.

    // [^\r\n\p{L}\p{N}]?\p{L}+
    if (!IsRN(m_text[0]) && !IsN(m_text[0])) {
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

    // Case 3: Numeric Character Class (3rd Alternative)
    // {1,3} matches the previous token between 1 and 3 times, as many times as possible, giving back as needed (greedy)
    // all other symbols as previously described.

    // \p{N}{1,3}
    if (IsN(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsN(m_text[i]) || (i > 2)) break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // Case 4: Custom Character Combination (4th Alternative)
    // * matches the previous token between zero and unlimited times, as many times as possible, giving back as needed
    // (greedy) all other symbols as previously described.

    // ?[^\s\p{L}\p{N}]+[\r\n]*
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (NotLNZ(m_text[1]))) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i])) break;
      }
      if (i < m_text.size() && IsRN(m_text[i])) {
        for (; i < m_text.size(); ++i) {
          if (!IsRN(m_text[i])) break;
        }
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
      if (i < m_text.size() && IsRN(m_text[i])) {
        for (; i < m_text.size(); ++i) {
          if (!IsRN(m_text[i])) break;
        }
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // Case 5: Custom Character Combination (5th Alternative)
    // all symbols as previously described.

    // \s*[\r\n]+
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
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
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

    // Case 5: Custom Character Combination (6th and 7th Alternative)
    // all symbols as previously described.

    // \s+(?!\S)|\s+
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

    return U"";
  }

  /*
  std::regex does not directly support std::u32string_view (which is what U"" literals produce).
  The std::regex class template is specialized for char and wchar_t types, but not for char32_t or char16_t.
  To work with Unicode strings in regex, we therefore convert the std::u32string_view to a std::wstring.

  Wide strings, or std::wstring, are used for supporting a large character set, including Unicode
  characters and characters from various languages.
  */

  // // Convert std::u32string_view to std::wstring
  // static std::wstring U2Wstring(const std::u32string_view& u32str) {
  //   std::wstring wstr;
  //   wstr.reserve(u32str.size()); // Reserve space to avoid multiple allocations

  //   for (char32_t codepoint : u32str) {
  //     if (codepoint <= 0xFFFF) {
  //       // Single UTF-16 code unit
  //       wstr.push_back(static_cast<wchar_t>(codepoint));
  //     } else if (codepoint <= 0x10FFFF) {
  //       // Surrogate pair
  //       codepoint -= 0x10000;
  //       wchar_t high_surrogate = static_cast<wchar_t>((codepoint >> 10) + 0xD800);
  //       wchar_t low_surrogate = static_cast<wchar_t>((codepoint & 0x3FF) + 0xDC00);
  //       wstr.push_back(high_surrogate);
  //       wstr.push_back(low_surrogate);
  //     } else {
  //       // Invalid code point for UTF-16
  //       ORTX_CXX_API_THROW("Invalid UTF-32 codepoint encountered", ORT_INVALID_ARGUMENT);
  //     }
  //   }

  //   return wstr;
  // }

  // // Convert std::wstring to std::u32string_view
  // std::u32string_view W2Ustring(const std::wstring& wstr) {
  //     std::u32string_view u32str;
  //     u32str.reserve(wstr.size());  // Reserve space to avoid multiple allocations

  //     for (wchar_t wc : wstr) {
  //         if (wc <= 0x7F) {
  //             // 1-byte character (ASCII)
  //             u32str.push_back(static_cast<char32_t>(wc));
  //         } else if (wc <= 0x7FF) {
  //             // 2-byte character
  //             char32_t ch = (static_cast<char32_t>(wc) & 0x07FF) | 0x0800;
  //             u32str.push_back(ch);
  //         } else if (wc <= 0xFFFF) {
  //             // 3-byte character
  //             char32_t ch = (static_cast<char32_t>(wc) & 0x0FFF) | 0xD800;
  //             u32str.push_back(ch);
  //             ch = (static_cast<char32_t>(wc) >> 10) | 0xDC00;
  //             u32str.push_back(ch);
  //         } else if (wc <= 0x10FFFF) {
  //             // 4-byte character (surrogate pairs)
  //             char32_t ch = ((wc >> 10) & 0x3FF) | 0xD800;
  //             u32str.push_back(ch);
  //             ch = (wc & 0x3FF) | 0xDC00;
  //             u32str.push_back(ch);
  //         } else {
  //             // Invalid Unicode code point
  //             ORTX_CXX_API_THROW("Invalid wide character encountered", ORT_INVALID_ARGUMENT);
  //         }
  //     }

  //     return u32str;
  // }

  // // Use the C++ standard library to handle regex pattern matching for compatible strings
  // std::u32string_view RegexMatchSTD(const std::u32string_view& regex) {
  //   std::wstring wstr = U2Wstring(m_text);
  //   std::wstring wpattern = U2Wstring(regex);

  //   std::wregex pattern(wpattern);
  //   std::wsmatch match;

  //   if (std::regex_search(wstr, match, pattern)) {
  //       std::u32string_view token = W2Ustring(match.str());
  //       m_text = std::u32string_view(match.suffix().first, match.suffix().second); // Update text to the remaining
  //       part after the match return token;
  //   } else {
  //       return U"";
  //   }
  // }

  // // Determine ufal::unilib::unicode regex category given string code.
  // static ufal::unilib::unicode::category_t StringToCategory(const std::string& category = "") {
  //   // Since C++ is not an interpreted language, we cannot simply convert the category to an object by typing
  //   // part of code into a string, so we manually parse it. Note that C++ also does not have switch-case statements.
  //   if (category == "C") {
  //     return ufal::unilib::unicode::C;
  //   } else if (category == "Cc") {
  //     return ufal::unilib::unicode::Cc;
  //   } else if (category == "Cf") {
  //     return ufal::unilib::unicode::Cf;
  //   } else if (category == "Cn") {
  //     return ufal::unilib::unicode::Cn;
  //   } else if (category == "Co") {
  //     return ufal::unilib::unicode::Co;
  //   } else if (category == "Cs") {
  //     return ufal::unilib::unicode::Cs;
  //   } else if (category == "L") {
  //     return ufal::unilib::unicode::L;
  //   } else if (category == "Ll") {
  //     return ufal::unilib::unicode::Ll;
  //   } else if (category == "Lm") {
  //     return ufal::unilib::unicode::Lm;
  //   } else if (category == "Lo") {
  //     return ufal::unilib::unicode::Lo;
  //   } else if (category == "Lt") {
  //     return ufal::unilib::unicode::Lt;
  //   } else if (category == "Lu") {
  //     return ufal::unilib::unicode::Lu;
  //   } else if (category == "M") {
  //     return ufal::unilib::unicode::M;
  //   } else if (category == "Mc") {
  //     return ufal::unilib::unicode::Mc;
  //   } else if (category == "Me") {
  //     return ufal::unilib::unicode::Me;
  //   } else if (category == "Mn") {
  //     return ufal::unilib::unicode::Mn;
  //   } else if (category == "N") {
  //     return ufal::unilib::unicode::N;
  //   } else if (category == "Nd") {
  //     return ufal::unilib::unicode::Nd;
  //   } else if (category == "Nl") {
  //     return ufal::unilib::unicode::Nl;
  //   } else if (category == "No") {
  //     return ufal::unilib::unicode::No;
  //   } else if (category == "P") {
  //     return ufal::unilib::unicode::P;
  //   } else if (category == "Pc") {
  //     return ufal::unilib::unicode::Pc;
  //   } else if (category == "Pd") {
  //     return ufal::unilib::unicode::Pd;
  //   } else if (category == "Pe") {
  //     return ufal::unilib::unicode::Pe;
  //   } else if (category == "Pf") {
  //     return ufal::unilib::unicode::Pf;
  //   } else if (category == "Pi") {
  //     return ufal::unilib::unicode::Pi;
  //   } else if (category == "Po") {
  //     return ufal::unilib::unicode::Po;
  //   } else if (category == "Ps") {
  //     return ufal::unilib::unicode::Ps;
  //   } else if (category == "S") {
  //     return ufal::unilib::unicode::S;
  //   } else if (category == "Sc") {
  //     return ufal::unilib::unicode::Sc;
  //   } else if (category == "Sk") {
  //     return ufal::unilib::unicode::Sk;
  //   } else if (category == "Sm") {
  //     return ufal::unilib::unicode::Sm;
  //   } else if (category == "So") {
  //     return ufal::unilib::unicode::So;
  //   } else if (category == "Z") {
  //     return ufal::unilib::unicode::Z;
  //   } else if (category == "Zl") {
  //     return ufal::unilib::unicode::Zl;
  //   } else if (category == "Zp") {
  //     return ufal::unilib::unicode::Zp;
  //   } else if (category == "Zs") {
  //     return ufal::unilib::unicode::Zs;
  //   } else {
  //     ORTX_CXX_API_THROW("Invalid category string provided!", ORT_INVALID_ARGUMENT);
  //   }
  // }

  // // Perform regex match given a list of categories (e.g. ?[\s\p{L}\p{N}]+), a premodifier, and a postmodifier
  // std::u32string_view RegexCategory(const std::vector<std::string>& categories, const std::string& premodifier = "",
  //                                   const std::string& postmodifier = "", const bool negated = false) {
  //   if (premodifier == "?") {
  //     // ?\p{_}+
  //     if (postmodifier == "+") {
  //       if ((m_text[0] == U' ') && (m_text.size() > 1) &&
  //           (negated ? !IsCategory(m_text[1], categories) : IsCategory(m_text[1], categories))) {
  //         size_t i = 2;
  //         for (; i < m_text.size(); ++i) {
  //           if ((negated ? IsCategory(m_text[i], categories) : !IsCategory(m_text[i], categories))) break;
  //         }
  //         std::u32string_view res = ustring(m_text.substr(0, i));
  //         m_text = m_text.substr(i);
  //         return res;
  //       }
  //       if ((negated ? !IsCategory(m_text[0], categories) : IsCategory(m_text[0], categories))) {
  //         size_t i = 1;
  //         for (; i < m_text.size(); ++i) {
  //           if ((negated ? IsCategory(m_text[i], categories) : !IsCategory(m_text[i], categories))) break;
  //         }
  //         std::u32string_view res = ustring(m_text.substr(0, i));
  //         m_text = m_text.substr(i);
  //         return res;
  //       }
  //     }
  //   } else if (postmodifier == "+" || postmodifier == "*") {
  //     if ((negated ? !IsCategory(m_text[0], categories) : IsCategory(m_text[0], categories))) {
  //       size_t i = 1;
  //       for (; i < m_text.size(); ++i) {
  //         if ((negated ? IsCategory(m_text[i], categories) : !IsCategory(m_text[i], categories))) break;
  //       }
  //       std::u32string_view res = ustring(m_text.substr(0, i));
  //       m_text = m_text.substr(i);
  //       return res;
  //     }
  //   } else if (premodifier == "" && postmodifier == "" &&
  //              (categories.size() == 1 ? (categories[0] != "A" && categories[0] != "AL" && categories[0] != "sS")
  //                                      : true)) {
  //     if ((negated ? !IsCategory(m_text[0], categories) : IsCategory(m_text[0], categories))) {
  //       std::u32string_view res = ustring(m_text.substr(0, 1));
  //       m_text = m_text.substr(1);
  //       return res;
  //     }
  //   }

  //   // \p{_}{x,y}
  //   if (postmodifier.size() == 5 && postmodifier[0] == '{' && postmodifier[4] == '}' &&
  //       postmodifier[2] == ',') {  // modifier syntax hardcoded atm for simplicity
  //     size_t x = postmodifier[1] - '0';
  //     size_t y = postmodifier[3] - '0';
  //     if (IsCategory(m_text[0], categories)) {
  //       size_t i = 1;
  //       for (; i < m_text.size(); ++i) {
  //         if (!IsCategory(m_text[i], categories) || (i >= y)) break;
  //       }
  //       if (i >= x) {
  //         std::u32string_view res = ustring(m_text.substr(0, i));
  //         m_text = m_text.substr(i);
  //         return res;
  //       }
  //     }
  //   }

  //   // The following cases handles English apostrophe endings, such as for her's, can't, you're, etc.
  //   // We have this hard-coded implementation included as it is very common and thereby we speed up
  //   // compute by handling these common cases.

  //   // 's|'t|'re|'ve|'m|'ll|'d (lowercase)
  //   if (categories.size() == 1 && categories[0] == "AL") {
  //     if ((m_text[0] == U'\'') && (m_text.size() > 1)) {
  //       if ((m_text[1] == U's') || (m_text[1] == U't') || (m_text[1] == U'm') || (m_text[1] == U'd')) {
  //         std::u32string_view res = ustring(m_text.substr(0, 2));
  //         m_text = m_text.substr(2);
  //         return res;
  //       }

  //       if (m_text.size() > 2) {
  //         if (((m_text[1] == U'r') && (m_text[2] == U'e')) || ((m_text[1] == U'v') && (m_text[2] == U'e')) ||
  //             ((m_text[1] == U'l') && (m_text[2] == U'l'))) {
  //           std::u32string_view res = ustring(m_text.substr(0, 3));
  //           m_text = m_text.substr(3);
  //           return res;
  //         }
  //       }
  //     }
  //   }

  //   // (?i:'s|'t|'re|'ve|'m|'ll|'d) (case insensitive)
  //   if (categories.size() == 1 && categories[0] == "A") {
  //     if ((m_text[0] == U'\'') && (m_text.size() > 1)) {
  //       if ((m_text[1] == U's') || (m_text[1] == U't') || (m_text[1] == U'm') || (m_text[1] == U'd') ||
  //           (m_text[1] == U'S') || (m_text[1] == U'T') || (m_text[1] == U'M') || (m_text[1] == U'D')) {
  //         std::u32string_view res = ustring(m_text.substr(0, 2));
  //         m_text = m_text.substr(2);
  //         return res;
  //       }

  //       if (m_text.size() > 2) {
  //         if ((((m_text[1] == U'r') || (m_text[1] == U'R')) && ((m_text[2] == U'e') || (m_text[2] == U'E'))) ||
  //             (((m_text[1] == U'v') || (m_text[1] == U'V')) && ((m_text[2] == U'e') || (m_text[2] == U'E'))) ||
  //             (((m_text[1] == U'l') || (m_text[1] == U'L')) && ((m_text[2] == U'l') || (m_text[2] == U'L')))) {
  //           std::u32string_view res = ustring(m_text.substr(0, 3));
  //           m_text = m_text.substr(3);
  //           return res;
  //         }
  //       }
  //     }
  //   }

  //   if (categories.size() == 1 && categories[0] == "sS") {
  //     if ((m_text.size() >= 1) && (IsZ(m_text[0]))) {
  //       size_t i = 1;
  //       for (; i < m_text.size(); ++i) {
  //         if (!IsZ(m_text[i])) break;
  //       }
  //       if ((i > 1) && (i != m_text.size())) {  //\s+(?!\S)
  //         i--;
  //         std::u32string_view res = ustring(m_text.substr(0, i));
  //         m_text = m_text.substr(i);
  //         return res;
  //       }
  //       // \s+
  //       std::u32string_view res = ustring(m_text.substr(0, i));
  //       m_text = m_text.substr(i);
  //       return res;
  //     }
  //   }

  //   return U"";
  // }

  // // Function to wrap standalone \p{...} categories
  // std::string WrapStandaloneCategories(const std::string& input) {
  //   std::string modified_input = input;
  //   std::regex standalone_category_regex(R"(\\p\{[A-Za-z]+\})");

  //   // Use std::sregex_iterator to find matches
  //   std::sregex_iterator iter(modified_input.begin(), modified_input.end(), standalone_category_regex);
  //   std::sregex_iterator end;

  //   // Accumulate modifications in new string
  //   std::string result;
  //   size_t last_pos = 0;

  //   bool not_standalone = false;

  //   while (iter != end) {
  //     size_t match_pos = iter->position();
  //     size_t match_length = iter->length();

  //     // Add portion before match
  //     result.append(modified_input, last_pos, match_pos - last_pos);

  //     // Update not_standalone based on the preceding characters
  //     for (size_t i = match_pos; i > 0; --i) {
  //       if (modified_input[i - 1] == ']') {
  //         not_standalone = false;  // Exit if closing bracket hit
  //         break;
  //       }
  //       if (modified_input[i - 1] == '[') {
  //         not_standalone = true;  // Opening bracket hit
  //         break;
  //       }
  //     }

  //     // Wrap category only if standalone
  //     if (!not_standalone) {
  //       result.append("[");
  //       result.append(iter->str());  // Add matched category
  //       result.append("]");
  //     } else {
  //       // Add original match
  //       result.append(iter->str());
  //     }

  //     last_pos = match_pos + match_length;
  //     ++iter;
  //   }

  //   // Add remaining portion of input string after last match
  //   result.append(modified_input, last_pos, modified_input.length() - last_pos);

  //   // Updated regex for special numerical case (e.g. "\\p{N}{1,3}")
  //   std::regex special_case_regex(R"(\\p\{([A-Za-z]+)\}\{(\d+,\d+)\})");
  //   result = std::regex_replace(result, special_case_regex, "[\\p{$1}]{$2}");

  //   return result;
  // }

  // std::string ReplaceString(std::string input, const std::string& target, const std::string& replacement) {
  //   size_t pos = 0;
  //   while ((pos = input.find(target, pos)) != std::string::npos) {
  //     input.replace(pos, target.length(), replacement);
  //     pos += replacement.length();  // Move past the replacement
  //   }

  //   return input;
  // }

  // struct CategorySet {
  //   std::vector<std::string> categories;
  //   std::string premodifier;
  //   std::string postmodifier;
  //   bool negated = false;
  // };

  // // Perform regex matching given any general regex string (not just GPT2 or Llama)
  // std::u32string_view RegexMatchGeneral(const std::string& regex_expr) {
  //   std::string target = "\\s+(?!\\S)|\\s+";
  //   std::string replacement = "\\p{sS}";

  //   std::string input = ReplaceString(regex_expr, target, replacement);
  //   input = std::regex_replace(input, std::regex("\\\\s"), "\\p{Z}");
  //   input = std::regex_replace(input, std::regex(R"(\(\?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d\)\?)"),
  //                              "[\\p{A}]");  // Apostrophe endings case insensitive
  //   input = std::regex_replace(input, std::regex(R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]))"),
  //                              "[\\p{A}]");  // variant of the above
  //   input = std::regex_replace(input, std::regex(R"('s\|'t\|'re\|'ve\|'m\|'ll\|'d)"),
  //                              "[\\p{AL}]");  // Apostrophe endings lowercase
  //   input = std::regex_replace(input, std::regex("\\\\r\\\\n"), "\\p{rn}");

  //   // Wrap standalone categories
  //   input = WrapStandaloneCategories(input);

  //   std::regex category_set_regex(R"((.?)\[\^?\\p\{[A-Za-z]+\}(\\p\{[A-Za-z]+\})*\](\*|\+|\{\d+,\d+\})?)");
  //   std::regex single_category_regex(R"(\\p\{([A-Za-z]+)\})");

  //   // Split the input string by '|'
  //   std::vector<std::string> parts;
  //   std::stringstream ss(input);
  //   std::string part;
  //   while (std::getline(ss, part, '|')) {
  //     parts.push_back(part);
  //   }

  //   // Get category sets and perform RegexCategory on each the sets of each part
  //   for (const auto& part : parts) {
  //     std::vector<CategorySet> category_sets;
  //     std::sregex_iterator sets_begin(part.begin(), part.end(), category_set_regex);
  //     std::sregex_iterator sets_end;

  //     // Iterate through this regex part and create category sets
  //     for (std::sregex_iterator i = sets_begin; i != sets_end; ++i) {
  //       std::smatch match = *i;
  //       CategorySet category_set;

  //       if (match.length(1) > 0) {
  //         category_set.premodifier = match.str(1);
  //       }
  //       if (match.length(3) > 0) {
  //         category_set.postmodifier = match.str(3);
  //       }

  //       std::string category_str = match.str();
  //       category_set.negated = category_str.find("[^") != std::string::npos;

  //       auto single_category_begin =
  //           std::sregex_iterator(category_str.begin(), category_str.end(), single_category_regex);
  //       auto single_category_end = std::sregex_iterator();

  //       for (std::sregex_iterator j = single_category_begin; j != single_category_end; ++j) {
  //         std::smatch single_category_match = *j;
  //         category_set.categories.push_back(single_category_match.str(1));
  //       }

  //       category_sets.push_back(category_set);
  //     }

  //     std::u32string_view m_text_copy = m_text;
  //     std::u32string_view concatenated = U"";
  //     bool skip = false;

  //     // Perform RegexCategory on each category set
  //     for (const auto& set : category_sets) {
  //       if (!skip) {
  //         std::u32string_view res = RegexCategory(set.categories, set.premodifier, set.postmodifier, set.negated);
  //         if (res != U"") {
  //           // concatenated.append(res);
  //           //  TODO: Fix concatenation
  //         } else if (set.postmodifier == "+" && (set.categories.size() == 1 ? (set.categories[0] != "rn") : true)) {
  //           // + requires a match of one or more, so if there is not at least one, the whole set has no match
  //           m_text = m_text_copy;
  //           skip = true;
  //         }
  //       }
  //     }

  //     if (concatenated != U"" && !skip) return concatenated;
  //   }

  //   return U"";
  // }

  // // RegexMatchCustom takes a regular expression 'regex_expr' and perform pattern matching to get the next token.
  // // If the regex can be parsed by our general RegexMatchGeneral parser designed to handle the majority of regex
  // cases
  // // it will be taken care of there. If not, we check to see if the regex can be parsed with the C++ Standard Library
  // // methods.
  // std::u32string_view RegexMatchCustom(const std::string& regex_expr) {
  //   try {
  //     if (regex_expr == GPT2_REGEX_PATTERN) {
  //       return RegexMatchGPT2();
  //     } else if (regex_expr == LLAMA_REGEX_PATTERN || regex_expr == LLAMA_REGEX_PATTERN_2) {
  //       return RegexMatchLlama3();
  //     } else {
  //       std::u32string_view res = RegexMatchGeneral(regex_expr);
  //       if (res != U"")
  //         return res;
  //       else
  //         ;
  //       // return RegexMatchSTD(ustring(regex_expr));
  //     }
  //   } catch (const std::exception& /* ex */) {
  //     std::string part1 = "Regex '";
  //     std::string part2 = "' not supported!";
  //     std::string msg = part1 + regex_expr + part2;
  //     ORTX_CXX_API_THROW(msg, ORT_INVALID_ARGUMENT);
  //   }

  //   return U"";
  // }

  static bool IsRN(char32_t ch) { return ch == U'\r' || ch == U'\n'; }

  // static bool IsCategory(char32_t ch, std::vector<std::string> categories) {
  //   if (std::find(categories.begin(), categories.end(), "rn") != categories.end() && IsRN(ch)) {
  //     return true;
  //   } else {
  //     categories.erase(std::remove(categories.begin(), categories.end(), "rn"), categories.end());
  //     auto cat = ufal::unilib::unicode::category(ch);
  //     for (auto str : categories) {
  //       ufal::unilib::unicode::category_t category = StringToCategory(str);
  //       if ((category & cat) != 0) {
  //         return true;
  //       }
  //     }
  //   }
  //   return false;
  // }

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
