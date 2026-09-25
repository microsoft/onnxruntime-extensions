// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#ifdef ENABLE_TF_STRING
#include "text/vector_to_string.hpp"
#include "text/string_to_vector.hpp"

// Test that VectorToStringImpl rejects a map attribute where a later line has
// more value columns than the first line (which would previously cause a heap
// OOB write in ParseValues).
TEST(VectorToStringTest, InconsistentLineWidthThrows) {
  // First line has 1 value column, second line has 16 — must be rejected.
  std::string map = "a\t1\nb\t1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16";
  std::string unk = "unk";
  EXPECT_THROW(VectorToStringImpl(map, unk), std::exception);
}

// Test that VectorToStringImpl rejects a map attribute where a later line has
// fewer value columns than the first line.
TEST(VectorToStringTest, FewerColumnsThrows) {
  // First line has 3 value columns, second line has 1.
  std::string map = "a\t1 2 3\nb\t4";
  std::string unk = "unk";
  EXPECT_THROW(VectorToStringImpl(map, unk), std::exception);
}

// Test that VectorToStringImpl succeeds with consistent line widths.
TEST(VectorToStringTest, ConsistentLineWidthsSucceeds) {
  std::string map = "a\t1 2 3\nb\t4 5 6\nc\t7 8 9";
  std::string unk = "unk";
  EXPECT_NO_THROW(VectorToStringImpl(map, unk));
}

// Test that StringToVectorImpl rejects a map attribute where a later line has
// more value columns than the first line.
TEST(StringToVectorTest, InconsistentLineWidthThrows) {
  // First line has 1 value column, second line has 16.
  std::string map = "a\t1\nb\t1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16";
  std::string unk = "0";
  EXPECT_THROW(StringToVectorImpl(map, unk), std::exception);
}

// Test that StringToVectorImpl rejects a map attribute where a later line has
// fewer value columns than the first line.
TEST(StringToVectorTest, FewerColumnsThrows) {
  // First line has 3 value columns, second line has 1.
  std::string map = "a\t1 2 3\nb\t4";
  std::string unk = "0 0 0";
  EXPECT_THROW(StringToVectorImpl(map, unk), std::exception);
}

// Test that StringToVectorImpl succeeds with consistent line widths.
TEST(StringToVectorTest, ConsistentLineWidthsSucceeds) {
  std::string map = "a\t1 2 3\nb\t4 5 6\nc\t7 8 9";
  std::string unk = "0 0 0";
  EXPECT_NO_THROW(StringToVectorImpl(map, unk));
}

#endif  // ENABLE_TF_STRING
