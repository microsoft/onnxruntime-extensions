// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test_sparse.h"
#include "sparse.hpp"

TEST(utils, create_ragged_from_dense) {
  std::vector<int64_t> shape{2, 3};
  std::vector<float> values{0, 1, 2, 3, 4, 5};
  SparseInTensor<float> tensor;
  SparseInTensor<float>::create_ragged_from_dense(shape, values, tensor);
  EXPECT_EQ(tensor.ndims(), 2);
  EXPECT_EQ(tensor.shape(), shape);
  EXPECT_EQ(tensor.size(), 84);
  EXPECT_EQ(tensor.nvalues(), 6);
  EXPECT_EQ(tensor.nindices(), 3);
  EXPECT_EQ(tensor.indices(), std::vector<int64_t>({0, 3, 6}));
  EXPECT_EQ(tensor.values(), values);
}

TEST(utils, create_ragged) {
  std::vector<int64_t> indices{0, 3, 6};
  std::vector<float> values{0, 1, 2, 3, 4, 5};
  SparseInTensor<float> tensor;
  SparseInTensor<float>::create_ragged(values, indices, tensor);
  EXPECT_EQ(tensor.ndims(), 2);
  EXPECT_EQ(tensor.shape(), std::vector<int64_t>({2, 0}));
  EXPECT_EQ(tensor.size(), 84);
  EXPECT_EQ(tensor.nvalues(), 6);
  EXPECT_EQ(tensor.nindices(), 3);
  EXPECT_EQ(tensor.indices(), std::vector<int64_t>({0, 3, 6}));
  EXPECT_EQ(tensor.values(), values);
}
