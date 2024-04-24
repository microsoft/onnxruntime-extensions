// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <locale>
#include "gtest/gtest.h"
#include "ocos.h"

#include "c_only_test.h"

TEST(CApiTest, ApiTest) {
  int ver = OrtxGetAPIVersion();
  EXPECT_GT(ver, 0);
  OrtxTokenizer* tokenizer = NULL;
  extError_t err = OrtxCreateTokenizer(&tokenizer, "data/tiktoken");
  EXPECT_EQ(err, kOrtxOK);

  const char* input = "This is a test";
  char* decoded_text = NULL;
  err = tokenize_text(tokenizer, input, &decoded_text);
  EXPECT_EQ(err, kOrtxOK);
  EXPECT_STREQ(decoded_text, input);
  free(decoded_text);
}
