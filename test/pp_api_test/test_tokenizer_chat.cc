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
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/models/phi-4");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxString> templated_text;
  std::string messages_json = R"(
    [
      {
        "role": "system",
        "content": "You are a helpful assistant.",
        "tools": "Calculator"
      },
      {
        "role": "user",
        "content": "How do I add two numbers?"
      },
      {
        "role": "assistant",
        "content": "You can add numbers by using the '+' operator."
      }
    ])";

  auto err = OrtxApplyChatTemplate(
    tokenizer.get(), nullptr,
    messages_json.c_str(), templated_text.ToBeAssigned(), true);

  if (err != kOrtxOK) {
    std::cout << "Failed to apply chat template, stopping the test." << std::endl;
    std::cout << "Error code: " << err << std::endl;
    std::cout << "Error message: " << OrtxGetLastErrorMessage() << std::endl;
  }

  const char* text = nullptr;
  OrtxStringGetCstr(templated_text.get(), &text);

  // From HuggingFace Python output for 'microsoft/Phi-4-multimodal-instruct'
  std::string expected_output = "<|system|>You are a helpful assistant.<|tool|>Calculator<|/tool|><|end|><|user|>"
    "How do I add two numbers?<|end|><|assistant|>You can add numbers by using the '+' operator.<|end|><|assistant|>";

  ASSERT_EQ(text, expected_output);
}
