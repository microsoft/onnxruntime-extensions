// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "../ocos/ortcustomops.h"
TEST(ortcustomlib, nops) {
  size_t count = NumberOfAvailableOperators();
  EXPECT_GT(count, 5);
  const char* name = GetNameOfAvailableOperator(0);
  EXPECT_EQ(std::string(name), std::string("NegPos"));
}
