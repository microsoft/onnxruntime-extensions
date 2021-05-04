// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include <dlib/matrix.h>

using namespace dlib;

TEST(math, matrix_op) {
  matrix<float> M(3,3);
  M = 54.2,  7.4,  12.1,
      1,     2,    3,
      5.9,   0.05, 1;

  matrix<float,3,1> y;
  y = 3.5,
      1.2,
      7.8;

  matrix<float> x = inv(M)*y;
  EXPECT_FLOAT_EQ(x(1, 0), -13.909741);
}
