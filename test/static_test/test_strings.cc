// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "string_utils.h"
#include "text/string_regex_split_re.hpp"

#ifdef ENABLE_RE2
using ort_regex = re2::RE2;
#else
using ort_regex = std::regex;
#endif

TEST(strings, regex_test) {

  std::vector<std::string> str_input = {"def myfunc():"};
  std::vector<std::string> str_pattern = {R"(def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):)"};
  std::vector<std::string> str_rewrite = {"static PyObject*\npy_$1(void)\n{"};

  bool global_replace_ = false;
  size_t size = str_input.size();

  std::regex reg(str_pattern[0]);

  if (global_replace_) {
    for (int64_t i = 0; i < size; i++) {
      std::cout << "Input:" << str_input[i] << " Pattern: " << str_pattern[0] << " Replace:" << str_rewrite[0] << std::endl;
      str_input[i] = std::regex_replace(str_input[i], reg, str_rewrite[0]);
      std::cout << "Output:" << str_input[i] << std::endl;
    }
  } else {
    for (int64_t i = 0; i < size; i++) {
      std::cout << "Input:" << str_input[i] << " Pattern: " << str_pattern[0] << " Replace:" << str_rewrite[0] << std::endl;
      str_input[i] = std::regex_replace(str_input[i], reg, str_rewrite[0], std::regex_constants::format_first_only);
      std::cout << "Output:" << str_input[i] << std::endl;
    }
  }
}

TEST(strings, regex_split) {
  std::string input = "hello  world";
  ort_regex reg("(\\s)");
  ort_regex keep_reg("\\s");
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
  ort_regex reg("(\\s)");
  ort_regex keep_reg("");
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

TEST(strings, regex_split_no_matched) {
  std::string input = "helloworld";
  ort_regex reg("(\\s)");
  ort_regex keep_reg("");
  std::vector<std::string_view> tokens;
  std::vector<int64_t> begin_offsets;
  std::vector<int64_t> end_offsets;
  RegexSplitImpl(input, reg, true, keep_reg, tokens, begin_offsets, end_offsets);
  std::vector<std::string_view> expected_tokens{"helloworld"};
  std::vector<int64_t> expected_begin_offsets{0};
  std::vector<int64_t> expected_end_offsets{10};
  EXPECT_EQ(expected_tokens, tokens);
  EXPECT_EQ(expected_begin_offsets, begin_offsets);
  EXPECT_EQ(expected_end_offsets, end_offsets);
}

TEST(strings, regex_split_begin_end_delim) {
  std::string input = " hello world ";
  ort_regex reg("(\\s)");
  ort_regex keep_reg("\\s");
  std::vector<std::string_view> tokens;
  std::vector<int64_t> begin_offsets;
  std::vector<int64_t> end_offsets;
  RegexSplitImpl(input, reg, true, keep_reg, tokens, begin_offsets, end_offsets);
  std::vector<std::string_view> expected_tokens{" ", "hello"," ", "world", " "};
  std::vector<int64_t> expected_begin_offsets{0, 1, 6, 7, 12};
  std::vector<int64_t> expected_end_offsets{1, 6, 7, 12, 13};
  EXPECT_EQ(expected_tokens, tokens);
  EXPECT_EQ(expected_begin_offsets, begin_offsets);
  EXPECT_EQ(expected_end_offsets, end_offsets);
}
