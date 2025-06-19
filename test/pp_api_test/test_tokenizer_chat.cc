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

  std::string expected_decoder_output = "systemYou are a helpful assistant."
                                        "userHow should I explain the Internet?"
                                        "assistant";

  ASSERT_EQ(std::string(text), expected_decoder_output);
}

TEST(OrtxTokenizerTest, Phi3MediumChatTemplate) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/phi-3-medium");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTensorResult> templated_text;
  std::string messages_json = R"(
    [
      {
        "role": "system",
        "content": "System message"
      },
      {
        "role": "user",
        "content": "Hello, can you call some tools for me?"
      },
      {
        "role": "assistant",
        "content": "Sure, I can calculate the sum for you!"
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

  // From HuggingFace Python output for 'microsoft/Phi-3-medium-4k-instruct'
  std::string expected_output = "<|user|>\nHello, can you call some tools for me?<|end|>\n<|assistant|>\n"
                                "Sure, I can calculate the sum for you!<|end|>\n";

  ASSERT_EQ(text_ptr, expected_output);

  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  const char* input[] = { text_ptr };
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  ASSERT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({32010, 15043, 29892, 508, 366, 1246, 777, 8492, 363, 592,
                                                29973, 32007, 32001, 18585, 29892, 306, 508, 8147, 278,
                                                2533, 363, 366, 29991, 32007}));
  
  OrtxObjectPtr<OrtxStringArray> decoded_text;
  OrtxDetokenize(tokenizer.get(), token_ids.get(), decoded_text.ToBeAssigned());
  EXPECT_EQ(decoded_text.Code(), kOrtxOK);

  const char* text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &text);

  std::string expected_decoder_output = "Hello, can you call some tools for me?Sure, I can calculate the sum for you!";

  ASSERT_EQ(std::string(text), expected_decoder_output);
}

TEST(OrtxTokenizerTest, Phi3MiniChatTemplate) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/phi-3-mini");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTensorResult> templated_text;
  std::string messages_json = R"(
    [
      {
        "role": "system",
        "content": "System message"
      },
      {
        "role": "user",
        "content": "Hello, can you call some tools for me?"
      },
      {
        "role": "assistant",
        "content": "Sure, I can calculate the sum for you!"
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

  // From HuggingFace Python output for 'microsoft/Phi-3-mini-4k-instruct'
  std::string expected_output = "<|system|>\nSystem message<|end|>\n<|user|>\nHello, can you call some tools for me?<|end|>\n"
                                "<|assistant|>\nSure, I can calculate the sum for you!<|end|>\n<|assistant|>\n";

  ASSERT_EQ(text_ptr, expected_output);

  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  const char* input[] = { text_ptr };
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  ASSERT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({32006, 2184, 2643, 32007, 32010, 15043, 29892, 508, 366, 1246, 777, 8492,
                                                363, 592, 29973, 32007, 32001, 18585, 29892, 306, 508, 8147, 278, 2533,
                                                363, 366, 29991, 32007, 32001}));
  
  OrtxObjectPtr<OrtxStringArray> decoded_text;
  OrtxDetokenize(tokenizer.get(), token_ids.get(), decoded_text.ToBeAssigned());
  EXPECT_EQ(decoded_text.Code(), kOrtxOK);

  const char* text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &text);

  std::string expected_decoder_output = "System messageHello, can you call some tools for me?Sure, I can calculate the sum for you!";

  ASSERT_EQ(std::string(text), expected_decoder_output);
}

TEST(OrtxTokenizerTest, Phi3VisionChatTemplate) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/phi-3-vision");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTensorResult> templated_text;
  std::string messages_json = R"(
    [
      {
        "role": "system",
        "content": "System message"
      },
      {
        "role": "user",
        "content": "Hello, can you call some tools for me?"
      },
      {
        "role": "assistant",
        "content": "Sure, I can calculate the sum for you!"
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

  // From HuggingFace Python output for 'microsoft/Phi-3-vision-128k-instruct'
  std::string expected_output = "<|system|>\nSystem message<|end|>\n<|user|>\nHello, can you call some tools for me?<|end|>\n"
                                "<|assistant|>\nSure, I can calculate the sum for you!<|end|>\n";

  ASSERT_EQ(text_ptr, expected_output);

  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  const char* input[] = { text_ptr };
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  ASSERT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({1, 32006, 29871, 13, 3924, 2643, 32007, 29871, 13, 32010, 29871, 13,
                                                10994, 29892, 508, 366, 1246, 777, 8492, 363, 592, 29973, 32007, 29871,
                                                13, 32001, 18585, 29892, 306, 508, 8147, 278, 2533, 363, 366, 29991,
                                                32007, 29871, 13}));
  
  OrtxObjectPtr<OrtxStringArray> decoded_text;
  OrtxDetokenize(tokenizer.get(), token_ids.get(), decoded_text.ToBeAssigned());
  EXPECT_EQ(decoded_text.Code(), kOrtxOK);

  const char* text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &text);

  std::string expected_decoder_output = "\nSystem message\n\nHello, can you call some tools for me?\nSure, I can calculate the sum for you!\n";

  ASSERT_EQ(std::string(text), expected_decoder_output);
}

TEST(OrtxTokenizerTest, Gemma3ChatTemplate) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/models/gemma-3");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTensorResult> templated_text;
  std::string messages_json = R"(
    [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "url": "https://huggingface.co/spaces/big-vision/paligemma-hf/resolve/main/examples/password.jpg"
          },
          {
            "type": "text",
            "text": "What is the password?"
          }
        ]
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

  // From HuggingFace Python output for 'google/gemma-3-4b-it'
  std::string expected_output = "<bos><start_of_turn>user\n<start_of_image>What is the password?<end_of_turn>\n<start_of_turn>model\n";

  ASSERT_EQ(text_ptr, expected_output);

  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  const char* input[] = { text_ptr };
  OrtxTokenizeWithOptions(tokenizer.get(), input, 1, token_ids.ToBeAssigned(), false);
  ASSERT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({2, 105, 2364, 107, 255999, 3689, 563, 506, 8918, 236881, 106, 107, 105, 4368, 107}));
  
  OrtxObjectPtr<OrtxStringArray> decoded_text;
  OrtxDetokenize(tokenizer.get(), token_ids.get(), decoded_text.ToBeAssigned());
  EXPECT_EQ(decoded_text.Code(), kOrtxOK);

  const char* text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &text);

  std::string expected_decoder_output = "user\nWhat is the password?\nmodel\n";

  ASSERT_EQ(std::string(text), expected_decoder_output);
}