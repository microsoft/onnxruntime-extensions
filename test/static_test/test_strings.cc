// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "string_utils.h"
#ifdef ENABLE_RE2_REGEX
#include "text/re2_strings/string_regex_split_re.hpp"
#endif
#include "text/string_ecmaregex_split.hpp"

TEST(strings, std_regex_test) {
  std::regex regex("[\u2700-\u27bf\U0001f650-\U0001f67f\U0001f600-\U0001f64f\u2600-\u26ff"
                    "\U0001f300-\U0001f5ff\U0001f900-\U0001f9ff\U0001fa70-\U0001faff"
                    "\U0001f680-\U0001f6ff]");

  std::string test =u8"abcde😀🔍🦑😁🔍🎉😂🤣";
  auto result = std::regex_replace(test, regex, "");
  std::cout << test << std::endl;
  std::cout << result << std::endl;
}

#ifdef ENABLE_RE2_REGEX
TEST(strings, regex_split) {
  std::string input = "hello  world";
  re2::RE2 reg("(\\s)");
  re2::RE2 keep_reg("\\s");
  std::vector<std::string_view> tokens;
  std::vector<int64_t> begin_offsets;
  std::vector<int64_t> end_offsets;
  RegexSplitImpl(input, reg, true, keep_reg, tokens, begin_offsets, end_offsets);
  std::vector<std::string_view> expected_tokens{"hello", " ", " ", "world"};
  std::vector<int64_t> expected_begin_offsets{0, 5, 6, 7};
  std::vector<int64_t> expected_end_offsets{5, 6, 7, 12};
  EXPECT_EQ(expected_tokens, tokens);
  EXPECT_EQ(expected_begin_offsets, begin_offsets);
  EXPECT_EQ(expected_end_offsets, end_offsets);
}

TEST(strings, regex_split_skip) {
  std::string input = "hello world";
  re2::RE2 reg("(\\s)");
  re2::RE2 keep_reg("");
  std::vector<std::string_view> tokens;
  std::vector<int64_t> begin_offsets;
  std::vector<int64_t> end_offsets;
  RegexSplitImpl(input, reg, true, keep_reg, tokens, begin_offsets, end_offsets);
  std::vector<std::string_view> expected_tokens{"hello", "world"};
  std::vector<int64_t> expected_begin_offsets{0, 6};
  std::vector<int64_t> expected_end_offsets{5, 11};
  EXPECT_EQ(expected_tokens, tokens);
  EXPECT_EQ(expected_begin_offsets, begin_offsets);
  EXPECT_EQ(expected_end_offsets, end_offsets);
}
#endif

TEST(strings, regex_split_no_matched) {
  std::string input = "helloworld";
  std::regex reg("(\\s)");
  std::regex keep_reg("");
  std::vector<std::string_view> tokens;
  std::vector<int64_t> begin_offsets;
  std::vector<int64_t> end_offsets;
  ECMARegexSplitImpl(input, reg, true, keep_reg, tokens, begin_offsets, end_offsets);
  std::vector<std::string_view> expected_tokens{"helloworld"};
  std::vector<int64_t> expected_begin_offsets{0};
  std::vector<int64_t> expected_end_offsets{10};
  EXPECT_EQ(expected_tokens, tokens);
  EXPECT_EQ(expected_begin_offsets, begin_offsets);
  EXPECT_EQ(expected_end_offsets, end_offsets);
}

TEST(strings, regex_split_begin_end_delim) {
  std::string input = " hello world ";
  std::regex reg("(\\s)");
  std::regex keep_reg("\\s");
  std::vector<std::string_view> tokens;
  std::vector<int64_t> begin_offsets;
  std::vector<int64_t> end_offsets;
  ECMARegexSplitImpl(input, reg, true, keep_reg, tokens, begin_offsets, end_offsets);
  std::vector<std::string_view> expected_tokens{" ", "hello"," ", "world", " "};
  std::vector<int64_t> expected_begin_offsets{0, 1, 6, 7, 12};
  std::vector<int64_t> expected_end_offsets{1, 6, 7, 12, 13};
  EXPECT_EQ(expected_tokens, tokens);
  EXPECT_EQ(expected_begin_offsets, begin_offsets);
  EXPECT_EQ(expected_end_offsets, end_offsets);
}
