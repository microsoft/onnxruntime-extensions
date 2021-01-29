// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test_base64.h"
#include "base64.h"

TEST(base64, encode_decode) {
  std::vector<uint8_t> raw{0, 1, 2, 0, 45, 7, 255, 4};
  std::string encoded;
  base64_encode(raw, encoded);
  std::vector<uint8_t> decoded;
  base64_decode(encoded, decoded);
  EXPECT_EQ(raw, decoded);
}

TEST(base64, encode_decode_single) {
  std::vector<std::string> expected = {"AA==", "AQ==", "Ag=="};
  for (int i = 1; i <= 256; ++i) {
    uint8_t b = (uint8_t)(i % 256);
    std::vector<uint8_t> raw{b};
    std::string encoded;
    base64_encode(raw, encoded);
    if (b < expected.size()) {
      EXPECT_EQ(expected[b], encoded);
    }
    std::vector<uint8_t> decoded;
    base64_decode(encoded, decoded);
    EXPECT_EQ(raw, decoded);
  }
}

TEST(base64, decode_encode) {
  std::vector<uint8_t> raw;
  std::string encoded("AAECAC0HAAQ=");
  base64_decode(encoded, raw);
  std::string encode2;
  base64_encode(raw, encode2);
  EXPECT_EQ(encoded, encode2);
}

TEST(base64, decode_false_length) {
  std::vector<uint8_t> raw;
  std::string encoded("AAAC0HAAQ=");
  bool r = base64_decode(encoded, raw);
  EXPECT_EQ(r, false);
}

TEST(base64, decode_false_wrong) {
  std::vector<uint8_t> raw;
  std::string encoded("AAECAC0HAA'=");
  bool r = base64_decode(encoded, raw);
  EXPECT_EQ(r, false);
}
