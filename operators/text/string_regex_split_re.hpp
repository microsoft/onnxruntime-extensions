// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <algorithm>
#ifdef ENABLE_RE2
#include "re2/re2.h"
#else
#include <regex>
#endif

#ifdef ENABLE_RE2
template <typename T>
void RegexSplitImpl(const std::string& input, const RE2& pattern,
                    bool include_delimiter, const RE2& include_delim_regex,
                    std::vector<std::string_view>& tokens,
                    std::vector<T>& begin_offsets,
                    std::vector<T>& end_offsets) {
  re2::StringPiece leftover(input.data());
  re2::StringPiece last_end = leftover;
  re2::StringPiece extracted_delim_token;

  // Keep looking for split points until we have reached the end of the input.
  while (RE2::FindAndConsume(&leftover, pattern, &extracted_delim_token)) {
    std::string_view token(last_end.data(),
                           extracted_delim_token.data() - last_end.data());
    bool has_non_empty_token = token.length() > 0;
    bool should_include_delim =
        include_delimiter && include_delim_regex.FullMatch(
                                 extracted_delim_token, include_delim_regex);
    last_end = leftover;

    // Mark the end of the previous token, only if there was something.
    if (has_non_empty_token) {
      tokens.push_back(std::string_view(token.data(), token.size()));
      // Mark the end of the last token
      begin_offsets.push_back(token.data() - input.data());
      end_offsets.push_back(token.data() + token.length() - input.data());
    }

    if (should_include_delim) {
      // If desired, include the deliminator as a token.
      tokens.push_back(std::string_view(extracted_delim_token.data(), extracted_delim_token.size()));
      // Mark the end of the token at the end of the beginning of the delimiter.
      begin_offsets.push_back(extracted_delim_token.data() - input.data());
      end_offsets.push_back(extracted_delim_token.data() +
                            extracted_delim_token.length() - input.data());
    }
  }

  // Close the last token.
  if (!leftover.empty()) {
    tokens.push_back(std::string_view(leftover.data(), leftover.size()));
    begin_offsets.push_back(leftover.data() - input.data());
    end_offsets.push_back(leftover.data() + leftover.length() - input.data());
  }
}
#else
template <typename T>
void RegexSplitImpl(const std::string& input, const std::regex& pattern,
                    bool include_delimiter, const std::regex& include_delim_regex,
                    std::vector<std::string_view>& tokens,
                    std::vector<T>& begin_offsets,
                    std::vector<T>& end_offsets) {
  std::smatch base_match;
  if (std::regex_match(input, base_match, pattern)) {
    // The first sub_match is the whole string; the next
    // sub_match is the first parenthesized expression.
    if (base_match.size() == 2) {
      std::ssub_match base_sub_match = base_match[1];
      std::string base = base_sub_match.str();
      std::cout << input << " has a base of " << base << '\n';
    }
  }}
#endif