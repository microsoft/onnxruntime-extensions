// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#ifdef ENABLE_RE2_REGEX
#include "re2/re2.h"
#endif
#include "nlohmann/json.hpp"
#include "string_utils.h"
#include "ustring.h"


TEST(utils, make_string) {
  std::string res = MakeString("a", "b", 0);
  EXPECT_EQ(res, "ab0");
}

#ifdef ENABLE_RE2_REGEX
TEST(utils, re2_basic) {
  re2::StringPiece piece("1234");
  re2::RE2 reg("[0-9]*");
}
#endif

TEST(utils, json) {
  nlohmann::json j;
  j.push_back("foo");
  EXPECT_EQ(j.size(), 1);
}

TEST(utils, split_string) {
  auto result = SplitString("a b c d e f", " ");
  EXPECT_EQ(result.size(), 6);

  // contain a space
  result = SplitString("ab cd ef  gh", " ");
  EXPECT_EQ(result.size(), 5);

  // contain a space
  result = SplitString("ab cd ef  gh", " ", true);
  EXPECT_EQ(result.size(), 4);

  result = SplitString("abcd", " ");
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "abcd");

  result = SplitString("eabc\nasbd", "\n");
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "eabc");

  result = SplitString("a\nb\n", "\n");
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[1], "b");

  // two seps
  result = SplitString("ea,bc\nas,bd", ",\n");
  EXPECT_EQ(result.size(), 4);
  EXPECT_EQ(result[1], "bc");
}

TEST(utils, utf8) {
  std::vector<std::string> srcs = {"abc", "ABCé", "中文"};
  std::vector<std::string> lowered = {"abc", "abcé", "中文"};
  for (size_t i = 0; i < srcs.size(); ++i) {
    std::string lower;
    lower = srcs[i];
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    EXPECT_EQ(lowered[i], lower);
  }
}