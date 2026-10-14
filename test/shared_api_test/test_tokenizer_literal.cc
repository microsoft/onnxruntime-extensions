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

}  // namespace
