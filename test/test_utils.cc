// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test_utils.h"
#include "utils.h"
#include "re2/re2.h"


TEST(utils, make_string) {
  std::string res = MakeString("a", "b", 0);
  EXPECT_EQ(res, "ab0");
}

TEST(utils, re2_basic){
  re2::StringPiece piece("1234");
  re2::RE2 reg("[0-9]*");
}
