// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "narrow.h"

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
          auto search_it = std::search(it, str.first.end(),
                                       std::boyer_moore_searcher(st.str.begin(), st.str.end()));
#endif
          if (search_it == str.first.end()) {
            new_split_res.emplace_back(u32string_view(
                                           str.first.data() + search_pos, str.first.size() - search_pos),
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

  std::pair<bool, std::u32string_view> GetNextToken(std::string & regex_expr) {
    while (!m_text.empty()) {
      auto res = RegexMatchCustom(regex_expr);
      if (res.empty()) {
        m_text = m_text.substr(1);
        continue;
      }
      return {true, res};
    }
    return {false, {}};
  }

  const std::string GPT2_REGEX_PATTERN = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
  const std::string LLAMA_REGEX_PATTERN_1 = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  const std::string LLAMA_REGEX_PATTERN_2 = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

 private:

  /*
  std::regex does not directly support std::u32string (which is what U"" literals produce).
  The std::regex class template is specialized for char and wchar_t types, but not for char32_t or char16_t.
  To work with Unicode strings, we therefore convert the std::u32string to a std::wstring.

  Wide strings, or std::wstring, are used for supporting a large character set, including Unicode
  characters and characters from various languages.
  */

  static std::wstring U2Wstring(const std::u32string& u32str) {
    std::wstring wstr;
    wstr.reserve(u32str.size()); // Reserve space to avoid multiple allocations

    for (char32_t codepoint : u32str) {
      if (codepoint <= 0xFFFF) {
        // Single UTF-16 code unit
        wstr.push_back(static_cast<wchar_t>(codepoint));
      } else if (codepoint <= 0x10FFFF) {
        // Surrogate pair
        codepoint -= 0x10000;
        wchar_t high_surrogate = static_cast<wchar_t>((codepoint >> 10) + 0xD800);
        wchar_t low_surrogate = static_cast<wchar_t>((codepoint & 0x3FF) + 0xDC00);
        wstr.push_back(high_surrogate);
        wstr.push_back(low_surrogate);
      } else {
        // Invalid code point for UTF-16
        throw std::runtime_error("Invalid UTF-32 codepoint encountered");
      }
    }

    return wstr;
  }


  std::u32string W2Ustring(const std::wstring& wstr) {
      std::u32string u32str;
      u32str.reserve(wstr.size());  // Reserve space to avoid multiple allocations

      for (wchar_t wc : wstr) {
          if (wc <= 0x7F) {
              // 1-byte character (ASCII)
              u32str.push_back(static_cast<char32_t>(wc));
          } else if (wc <= 0x7FF) {
              // 2-byte character
              char32_t ch = (static_cast<char32_t>(wc) & 0x07FF) | 0x0800;
              u32str.push_back(ch);
          } else if (wc <= 0xFFFF) {
              // 3-byte character
              char32_t ch = (static_cast<char32_t>(wc) & 0x0FFF) | 0xD800;
              u32str.push_back(ch);
              ch = (static_cast<char32_t>(wc) >> 10) | 0xDC00;
              u32str.push_back(ch);
          } else if (wc <= 0x10FFFF) {
              // 4-byte character (surrogate pairs)
              char32_t ch = ((wc >> 10) & 0x3FF) | 0xD800;
              u32str.push_back(ch);
              ch = (wc & 0x3FF) | 0xDC00;
              u32str.push_back(ch);
          } else {
              // Invalid Unicode code point
              throw std::runtime_error("Invalid wide character encountered");
          }
      }

      return u32str;
  }

  // Use the C++ standard library to handle regex pattern matching for compatible strings
  std::u32string_view RegexMatchSTD(const std::u32string& regex) {
    static std::u32string text(m_text);
    std::wstring wstr = U2Wstring(text);
    std::wstring wpattern = U2Wstring(regex);

    std::wregex pattern(wpattern);
    std::wsmatch match;

    if (std::regex_search(wstr, match, pattern)) {
        std::u32string temp = W2Ustring(match.str());
        std::u32string_view token = std::u32string_view(temp.data(), match.str().size());
        m_text = std::u32string(match.suffix().first, match.suffix().second); // Update text to the remaining part after the match
        return token;
    } else {
        return std::u32string_view{};
    }
  }

  // For efficiency, we handle manual parsing for certain popular regex strings commonly used in popular models,
  // such as GPT2 and Llama3.
  // Reference: https://github.com/ggerganov/llama.cpp/blob/9fe94ccac92693d4ae1bc283ff0574e8b3f4e765/src/unicode.cpp#L225

  // GPT2 Python Regex pattern:
  // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

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

    // Case 2: Language Character Class (8th Alternative)
    // ? matches the previous token between zero and one times, as many times as possible, giving back as needed (greedy)
    // \p{L} matches any kind of letter from any language
    // + matches the previous token between one and unlimited times, as many times as possible, giving back as needed (greedy)
    
    // ?\p{L}+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && IsL(m_text[1])) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if (!IsL(m_text[i]))
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (IsL(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsL(m_text[i]))
          break;
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
        if (!IsN(m_text[i]))
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (IsN(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsN(m_text[i]))
          break;
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

    return std::u32string_view{};
  }

  // Llama3 Python Regex pattern:
  // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  std::u32string_view RegexMatchLlama3() {

    // Case 1: English apostrophe handling, case insensitive (1st Alternative, the endings for her's, CAN'T, etc.)
    // (?_: ...) is a Non-capturing Group, which matches the tokens contained with the effective flags following ?
    // i modifier performs a case insensitive match (ignores case of [a-zA-Z])
    // all other symbols as previously described.

    // Note: the sequencial of the following if should not be switched, which follows the python regex's syntax

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if ((m_text[0] == U'\'') && (m_text.size() > 1)) {
      if ((m_text[1] == U's') || (m_text[1] == U't') ||
          (m_text[1] == U'm') || (m_text[1] == U'd') ||
          (m_text[1] == U'S') || (m_text[1] == U'T') ||
          (m_text[1] == U'M') || (m_text[1] == U'D')) {
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
    if (!IsRN(m_text[0]) && !IsN(m_text[0])){
      if (IsL(m_text[0]) || ((m_text.size() > 1) && IsL(m_text[1]))){
        size_t i = 1;
        for (; i < m_text.size(); ++i) {
          if (!IsL(m_text[i]))
            break;
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
    if (IsN(m_text[0])){
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsN(m_text[i]) || (i > 2))
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // Case 4: Custom Character Combination (4th Alternative)
    // * matches the previous token between zero and unlimited times, as many times as possible, giving back as needed (greedy)
    // all other symbols as previously described.

    // ?[^\s\p{L}\p{N}]+[\r\n]*
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (NotLNZ(m_text[1]))) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i]))
          break;
      }
      if (i < m_text.size() && IsRN(m_text[i])){
        for (; i < m_text.size(); ++i) {
          if (!IsRN(m_text[i]))
            break;
        }
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
      if (i < m_text.size() && IsRN(m_text[i])){
        for (; i < m_text.size(); ++i) {
          if (!IsRN(m_text[i]))
            break;
        }
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // Case 5: Custom Character Combination (5th Alternative)
    // all symbols as previously described.

    // \s*[\r\n]+
    if (IsZ(m_text[0])){
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsZ(m_text[i]))
          break;
      }
      if (i < m_text.size() && IsRN(m_text[i])){
        for (; i < m_text.size(); ++i) {
          if (!IsRN(m_text[i]))
            break;
        }
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (IsRN(m_text[0])){
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsRN(m_text[i]))
          break;
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

    return std::u32string_view{};
  }

  // RegexMatchCustom takes a regular expression 'regex_expr' and perform pattern matching to get the next token.
  // If the regex is familiar and matches that for one of our pre-written parsers for GPT2 or Llama3, we use the
  // appropriate methods. If not, we check to see if the regex can be parsed with the C++ Standard Library methods.
  std::u32string_view RegexMatchCustom(const std::string & regex_expr) {
    std::vector<size_t> bpe_offsets;

    if (regex_expr == GPT2_REGEX_PATTERN) {
        return RegexMatchGPT2();
    } else if (regex_expr == LLAMA_REGEX_PATTERN_1 ||
               regex_expr == LLAMA_REGEX_PATTERN_2) {

        return RegexMatchLlama3();
    }

    try {
      return RegexMatchSTD(ustring(regex_expr));
    } catch (const std::exception& ex) {
      std::string part1 = "Regex '";
      std::string part2 = "' not supported!";
      throw std::runtime_error(part1 + regex_expr + part2);
    }

    return std::u32string_view{};
  }

  static bool IsRN(char32_t ch) {
    return ch == U'\r' || ch == U'\n';
  }

  static bool IsL(char32_t ch) {
    auto category = ufal::unilib::unicode::category(ch);
    return (category & ufal::unilib::unicode::L) != 0;
  }

  static bool IsN(char32_t ch) {
    auto category = ufal::unilib::unicode::category(ch);
    return (category & ufal::unilib::unicode::N) != 0;
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

}  // namespace bpe
}  // namespace ort_extensions
