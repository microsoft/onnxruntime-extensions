// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "ortx_cpp_helper.h"
#include "ortx_tokenizer.h"

namespace {

using ort_extensions::OrtxObjectPtr;

TEST(TokenizerSharedApiTest, LiteralEncodingIsAvailableThroughTheSharedLibrary) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/phi-3");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
  const char* input[] = {"Hello <|assistant|> world"};
  OrtxObjectPtr<OrtxTokenId2DArray> sequences;
  ASSERT_EQ(OrtxTokenizeLiteral(tokenizer.get(), input, 1, sequences.ToBeAssigned()), kOrtxOK)
      << OrtxGetLastErrorMessage();
  const extTokenId_t* ids = nullptr;
  size_t count = 0;
  ASSERT_EQ(OrtxTokenId2DArrayGetItem(sequences.get(), 0, &ids, &count), kOrtxOK);
  ASSERT_NE(ids, nullptr);
  ASSERT_GT(count, 0u);
  const std::vector<extTokenId_t> tokens(ids, ids + count);
  EXPECT_EQ(std::find(tokens.begin(), tokens.end(), 32001), tokens.end());
}

TEST(TokenizerSharedApiTest, BoundedEncodingIsAvailableThroughTheSharedLibrary) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/phi-2");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK);
  ASSERT_EQ(OrtxPrepareBoundedTokenization(tokenizer.get()), kOrtxOK) << OrtxGetLastErrorMessage();
  const char* input[] = {"return x + 12345"};
  OrtxObjectPtr<OrtxTokenId2DArray> sequences;
  ASSERT_EQ(OrtxTokenize(tokenizer.get(), input, 1, sequences.ToBeAssigned()), kOrtxOK);
  const extTokenId_t* expected{};
  size_t expected_count{};
  ASSERT_EQ(OrtxTokenId2DArrayGetItem(sequences.get(), 0, &expected, &expected_count), kOrtxOK);
  std::vector<extTokenId_t> output(32, 0xDEADBEEF);
  size_t count = 99;
  EXPECT_NE(OrtxTokenizeIntoBuffer(tokenizer.get(), input[0], 0, output.data(), output.size(), &count), kOrtxOK);
  EXPECT_EQ(count, 0u);
  EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](auto id) { return id == 0xDEADBEEF; }));
  ASSERT_EQ(OrtxTokenizeIntoBuffer(tokenizer.get(), input[0], 4096, output.data(), output.size(), &count), kOrtxOK);
  ASSERT_EQ(count, expected_count);
  EXPECT_TRUE(std::equal(output.begin(), output.begin() + count, expected));
}

}  // namespace
