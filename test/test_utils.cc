// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "../ocos/utils.h"

TEST(utils, make_string) {
  std::string res = MakeString("a", "b", 0);
  EXPECT_EQ(res, "ab0");
}
