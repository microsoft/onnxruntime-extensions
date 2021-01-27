// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test_utils.h"
#include "utils.h"
#include "re2/re2.h"
#include "nlohmann/json.hpp"

TEST(utils, make_string) {
  std::string res = MakeString("a", "b", 0);
  EXPECT_EQ(res, "ab0");
}

TEST(utils, re2_basic) {
  re2::StringPiece piece("1234");
  re2::RE2 reg("[0-9]*");
}

TEST(utils, json) {
  nlohmann::json j;
  j.push_back("foo");
  EXPECT_EQ(j.size(), 1);
}
