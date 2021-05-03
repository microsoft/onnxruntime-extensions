// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include <dlib/matrix.h>

using namespace dlib;

TEST(math, matrix_op) {
    matrix<double,3,1> y;
    matrix<double,3,3> M;
    matrix<double> x;

    // set all elements to 1
    y = 1;
    M = 1;

    x = y + y;
}

