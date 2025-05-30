// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <fstream>
#include <locale>
#include <algorithm>
#include "gtest/gtest.h"

#include "c_only_test.h"
#include "ortx_cpp_helper.h"

using namespace ort_extensions;

TEST(OrtxTokenizerTest, Phi4ChatTemplate) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/phi-4-base");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTensorResult> templated_text;
  std::string messages_json = R"(
    [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "How should I explain the Internet?"
      }
    ])";

  auto err = OrtxApplyChatTemplate(
    tokenizer.get(), nullptr,
    messages_json.c_str(), nullptr, templated_text.ToBeAssigned(), true, false);

  if (err != kOrtxOK) {
    std::cout << "Failed to apply chat template, stopping the test." << std::endl;
    std::cout << "Error code: " << err << std::endl;
    std::cout << "Error message: " << OrtxGetLastErrorMessage() << std::endl;
  }

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(templated_text.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(tensor.Code(), kOrtxOK) << "Failed to get tensor from templated text, stopping the test.";
  const char* text_ptr = nullptr;
  OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr);

  // From HuggingFace Python output for 'microsoft/Phi-4'
  std::string expected_output = "<|im_start|>system<|im_sep|>You are a helpful assistant.<|im_end|>"
                                "<|im_start|>user<|im_sep|>How should I explain the Internet?<|im_end|>"
                                "<|im_start|>assistant<|im_sep|>";

  ASSERT_EQ(text_ptr, expected_output);

  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  const char* input[] = { text_ptr };
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  ASSERT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({100264, 9125, 100266, 2675, 527, 264, 11190, 18328, 13,
                                                100265, 100264, 882, 100266, 4438, 1288, 358, 10552, 279,
                                                8191, 30, 100265, 100264, 78191, 100266}));
  
  OrtxObjectPtr<OrtxStringArray> decoded_text;
  OrtxDetokenize(tokenizer.get(), token_ids.get(), decoded_text.ToBeAssigned());
  EXPECT_EQ(decoded_text.Code(), kOrtxOK);

  const char* text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &text);

  EXPECT_STREQ(text, text_ptr);
}
