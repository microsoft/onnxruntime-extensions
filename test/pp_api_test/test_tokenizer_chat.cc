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

TEST(OrtxTokenizerTest, Phi4SpecialChatTemplate) {
  // Create tokenizer with options (skip_special_tokens = false)
  const char* option_keys[] = { "skip_special_tokens" };
  const char* option_values[] = { "false" };
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizerWithOptions, "data/phi-4-base", option_keys, option_values, 1);
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

  // Test detokenization with skip_special_tokens = false (tokens like <|im_start|> should be part of the output)
  OrtxDetokenize1D(tokenizer.get(), &ids_vec.front(), ids_vec.size(), decoded_text.ToBeAssigned());
  const char* special_text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &special_text);

  std::string expected_special_decoder_output = "<|im_start|>system<|im_sep|>You are a helpful assistant.<|im_end|><|im_start|>user<|im_sep|>How should I explain the Internet?<|im_end|><|im_start|>assistant<|im_sep|>";

  ASSERT_EQ(std::string(special_text), expected_special_decoder_output);
}

/*

Test loading chat template from chat_template.jinja file. The tokenizer_config.json from
the model files in data/phi-4-mini-reasoning has been modified and the "chat_template" attribute
has been removed from it. This is because the HuggingFace Transformers standard has changed and
the chat template is now stored as a separate file, not as a key in tokenizer_config.json, so if
a user calls load_pretrained or save_pretrained in the HF Python API, after loading a tokenizer, e.g.:

tokenizer = AutoTokenizer.from_pretrained(input_folder)
tokenizer.save_pretrained(output_folder)

There will be a difference in the tokenizer_config.json files, removing the chat_template key,
and adding a separate chat_template.jinja file with its value. This logic is now handled in
our backend, and tested below.

*/

TEST(OrtxTokenizerTest, Phi4MiniReasoningChatTemplate) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/phi-4-mini-reasoning");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTensorResult> templated_text;
  std::string messages_json = R"(
    [
      {
        "role": "system",
        "content": "You are a medieval knight and must provide explanations to modern people."
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

  // From HuggingFace Python output for 'microsoft/Phi-4-mini-reasoning'
  std::string expected_output = "<|system|>Your name is Phi, an AI math expert developed by Microsoft. "
                                "You are a medieval knight and must provide explanations to modern people."
                                "<|end|><|user|>How should I explain the Internet?<|end|><|assistant|>";

  ASSERT_EQ(text_ptr, expected_output);

  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  const char* input[] = { text_ptr };
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  ASSERT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({200022, 9719, 1308, 382, 102143, 11, 448, 20837, 13324, 8333,
                                                9742, 656, 8321, 13, 1608, 553, 261, 55145, 105457, 326,
                                                2804, 3587, 69457, 316, 6809, 1665, 13, 200020, 200021, 5299,
                                                1757, 357, 16644, 290, 7380, 30, 200020, 200019}));
  
  OrtxObjectPtr<OrtxStringArray> decoded_text;
  OrtxDetokenize(tokenizer.get(), token_ids.get(), decoded_text.ToBeAssigned());
  EXPECT_EQ(decoded_text.Code(), kOrtxOK);

  const char* text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &text);

  std::string expected_decoder_output = "Your name is Phi, an AI math expert developed by Microsoft. "
                                        "You are a medieval knight and must provide explanations to modern people."
                                        "How should I explain the Internet?";

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
  // Create tokenizer with options (add_special_tokens = false)
  const char* option_keys[] = { "add_special_tokens" };
  const char* option_values[] = { "false" };
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizerWithOptions, "data/models/gemma-3", option_keys, option_values, 1);
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

  // Test tokenization with add_special_tokens = false (options already set during tokenizer creation)
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
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

  // Set skip_special_tokens to false before the 1D detokenization
  const char* keys[] = {"skip_special_tokens"};
  const char* vals[] = {"false"};
  auto err2 = OrtxUpdateTokenizerOptions(tokenizer.get(), keys, vals, 1);
  ASSERT_EQ(err2, kOrtxOK);

  // Test detokenization with skip_special_tokens = false
  OrtxDetokenize1D(tokenizer.get(), &ids_vec.front(), ids_vec.size(), decoded_text.ToBeAssigned());
  const char* special_text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &special_text);

  std::string expected_special_decoder_output = "user\nWhat is the password?\nmodel\n";

  ASSERT_EQ(std::string(special_text), expected_special_decoder_output);
}

/*

Similar to the Phi4MiniReasoningChatTemplate test, this test checks for chat template loading
from a separate chat_template.json file, instead of a key in tokenizer_config.json. Although
the official HuggingFace model files for the gemma-3 model contain both the chat_template.json
file as well as the key in tokenizer_config.json, we have removed the key in "data/gemma-3-chat"
to test the special loading from the chat_template.json file.

*/

TEST(OrtxTokenizerTest, Gemma3SpecialChatTemplate) {
  // Create tokenizer with options (both add_special_tokens and skip_special_tokens = false)
  const char* option_keys[] = { "add_special_tokens", "skip_special_tokens" };
  const char* option_values[] = { "false", "False" }; // also testing case as this may also be used with our Python API
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizerWithOptions, "data/gemma-3-chat", option_keys, option_values, 2);
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

  // Test tokenization with add_special_tokens = false (options already set during tokenizer creation)
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
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

  // Test detokenization with skip_special_tokens = false (options already set during tokenizer creation)
  OrtxDetokenize1D(tokenizer.get(), &ids_vec.front(), ids_vec.size(), decoded_text.ToBeAssigned());
  const char* special_text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &special_text);

  std::string expected_special_decoder_output = "user\nWhat is the password?\nmodel\n";

  ASSERT_EQ(std::string(special_text), expected_special_decoder_output);
}

/*

Neither OpenAI nor HuggingFace have explicit Whisper chat template functionality.

OpenAI does not expose Whisper via a chat interface like ChatCompletion.
Instead, their Whisper API uses raw audio uploads. For finetuning or embedding Whisper
in pipelines, they rely on pre-tokenized sequences. They don’t use Jinja2 or chat templates
for Whisper at all — it is purely sequence input with prepended tokens like
<|startoftranscript|> manually inserted.

In HuggingFace transformers, for Whisper, you are expected to pass in the template manually
or have it defined in tokenizer_config.json.

However, the expected logic (same for HF and OAI) should emulate concatenating
message['content'], with no roles, separators, etc. We thereby automatically handle this
in ORT Extensions as well.

*/

TEST(OrtxTokenizerTest, WhisperChatTemplate) {
  // Create tokenizer with options (add_special_tokens = false and skip_special_tokens = true)
  const char* option_keys[] = { "add_special_tokens", "skip_special_tokens" };
  const char* option_values[] = { "false", "true" };
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizerWithOptions, "data/tokenizer/whisper.tiny", option_keys, option_values, 2);
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTensorResult> templated_text;

  std::string messages_json = R"(
    [
      {
        "role": "system",
        "content": "You are an audio assistant."
      },
      {
        "role": "user",
        "content": "transcribe this clip:"
      },
      {
        "role": "user",
        "content": "<|audio|>…binary…<|endofaudio|>"
      }
    ])";

  auto err = OrtxApplyChatTemplate(
    tokenizer.get(), nullptr,
    messages_json.c_str(), nullptr,
    templated_text.ToBeAssigned(), /*add_generation_prompt=*/false, /*tokenize=*/false);

  if (err != kOrtxOK) {
    std::cout << "Failed to apply Whisper chat template, stopping the test." << std::endl;
    std::cout << "Error code: " << err << std::endl;
    std::cout << "Error message: " << OrtxGetLastErrorMessage() << std::endl;
  }

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(templated_text.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(tensor.Code(), kOrtxOK) << "Failed to get tensor from templated text, stopping the test.";

  const char* text_ptr = nullptr;
  OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr);

  std::string expected_output = "You are an audio assistant.\ntranscribe this clip:\n<|audio|>…binary…<|endofaudio|>";

  ASSERT_EQ(std::string(text_ptr), expected_output);

  // Tokenize to confirm token IDs
  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  const char* input[] = { text_ptr };
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  ASSERT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  // Validate against HuggingFace Input IDs
  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({3223, 366, 364, 6278, 10994, 13, 198, 24999, 8056, 341, 7353, 25, 198, 27, 91, 46069, 91, 29, 1260, 48621, 1260, 27, 91, 521, 2670, 46069, 91, 29}));

  // Detokenize to confirm behavior
  OrtxObjectPtr<OrtxStringArray> decoded_text;
  OrtxDetokenize(tokenizer.get(), token_ids.get(), decoded_text.ToBeAssigned());
  EXPECT_EQ(decoded_text.Code(), kOrtxOK);

  const char* roundtrip_text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &roundtrip_text);

  ASSERT_EQ(std::string(roundtrip_text), expected_output);

  // Flip skip_special_tokens to false before the 1D detokenization
  const char* keys[] = {"skip_special_tokens"};
  const char* vals[] = {"false"};
  auto err2 = OrtxUpdateTokenizerOptions(tokenizer.get(), keys, vals, 1);
  ASSERT_EQ(err2, kOrtxOK);

  // Test detokenization with skip_special_tokens = false
  OrtxDetokenize1D(tokenizer.get(), &ids_vec.front(), ids_vec.size(), decoded_text.ToBeAssigned());
  const char* special_text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &special_text);

  std::string expected_special_decoder_output = "You are an audio assistant.\ntranscribe this clip:\n<|audio|>\xE2\x80\xA6" "binary\xE2\x80\xA6<|endofaudio|>";

  ASSERT_EQ(std::string(special_text), expected_special_decoder_output);
}

TEST(OrtxTokenizerTest, Qwen3ChatTemplate) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/qwen3");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTensorResult> templated_text;
  std::string messages_json = R"(
    [
      {
        "role": "user",
        "content": "Hi, this is a test!"
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

  // From HuggingFace Python output for 'Qwen/Qwen3-0.6B'
  std::string expected_output = "<|im_start|>user\nHi, this is a test!<|im_end|>\n<|im_start|>assistant\n";

  ASSERT_EQ(text_ptr, expected_output);

  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  const char* input[] = { text_ptr };
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  ASSERT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({151644, 872, 198, 13048, 11, 419, 374, 264, 1273, 0, 151645, 198, 151644, 77091, 198}));
  
  OrtxObjectPtr<OrtxStringArray> decoded_text;
  OrtxDetokenize(tokenizer.get(), token_ids.get(), decoded_text.ToBeAssigned());
  EXPECT_EQ(decoded_text.Code(), kOrtxOK);

  const char* text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &text);

  std::string expected_decoder_output = "user\nHi, this is a test!\nassistant\n";

  ASSERT_EQ(std::string(text), expected_decoder_output);
}

TEST(OrtxTokenizerTest, Phi4MiniChatTemplateWithMinjaTools) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/phi-4-mini");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTensorResult> templated_text;
  std::string messages_json = R"(
    [
      {
        "role": "system",
        "content": "You are a helpful assistant.",
        "tools": "[{\"name\": \"get_horoscope\", \"description\": \"Get today's horoscope for an astrological sign.\", \"parameters\": {\"sign\": {\"type\": \"str\", \"description\": \"An astrological sign like Taurus or Aquarius\", \"default\": \"\"}}}]"
      },
      {
        "role": "user",
        "content": "How should I explain the Internet?"
      }
    ])";

  auto err = OrtxApplyChatTemplate(
    tokenizer.get(), nullptr,
    messages_json.c_str(), nullptr,
    templated_text.ToBeAssigned(), true, false);

  if (err != kOrtxOK) {
    std::cout << "Failed to apply chat template (Minja/Phi-4 tools)" << std::endl;
    std::cout << "Error code: " << err << std::endl;
    std::cout << "Error message: " << OrtxGetLastErrorMessage() << std::endl;
  }

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(templated_text.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(tensor.Code(), kOrtxOK);

  const char* text_ptr = nullptr;
  OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr);

  std::string expected_output = "<|system|>You are a helpful assistant.<|tool|>"
                                "[{\"name\": \"get_horoscope\", \"description\": \"Get today's horoscope for an astrological sign.\", "
                                "\"parameters\": {\"sign\": {\"type\": \"str\", \"description\": \"An astrological sign like Taurus or Aquarius\", "
                                "\"default\": \"\"}}}]<|/tool|><|end|><|user|>How should I explain the Internet?<|end|><|assistant|>";

  ASSERT_EQ(std::string(text_ptr), expected_output);
}

TEST(OrtxTokenizerTest, Phi4MiniChatTemplateWithOAIToolType) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/phi-4-mini");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTensorResult> templated_text;
  std::string messages_json = R"(
    [
      {
        "role": "system",
        "content": "You are a helpful assistant.",
        "tools": "[{\"type\": \"tool\", \"name\": \"get_horoscope\", \"description\": \"Get today's horoscope for an astrological sign.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"sign\": {\"type\": \"string\", \"description\": \"An astrological sign like Taurus or Aquarius\"}}}}]"
      },
      {
        "role": "user",
        "content": "How should I explain the Internet?"
      }
    ])";

  auto err = OrtxApplyChatTemplate(
    tokenizer.get(), nullptr,
    messages_json.c_str(), nullptr,
    templated_text.ToBeAssigned(), true, false);

  if (err != kOrtxOK) {
    std::cout << "Failed to apply chat template (Minja/Phi-4 tools)" << std::endl;
    std::cout << "Error code: " << err << std::endl;
    std::cout << "Error message: " << OrtxGetLastErrorMessage() << std::endl;
  }

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(templated_text.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(tensor.Code(), kOrtxOK);

  const char* text_ptr = nullptr;
  OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr);

  std::string expected_output = "<|system|>You are a helpful assistant.<|tool|>"
                                "[{\"name\": \"get_horoscope\", \"description\": \"Get today's horoscope for an astrological sign.\", "
                                "\"parameters\": {\"sign\": {\"type\": \"str\", \"description\": \"An astrological sign like Taurus or Aquarius\"}}}]"
                                "<|/tool|><|end|><|user|>How should I explain the Internet?<|end|><|assistant|>";

  ASSERT_EQ(std::string(text_ptr), expected_output);
}

TEST(OrtxTokenizerTest, Phi4MiniChatTemplateWithOAIFunctionType) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/phi-4-mini");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTensorResult> templated_text;
  std::string messages_json = R"(
  [
    {
      "role": "system",
      "content": "You are a helpful assistant.",
      "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the current weather for a given location.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"The name of the city or location.\"}}, \"required\": [\"location\"]}}}, {\"type\": \"function\", \"function\": {\"name\": \"get_tourist_attractions\", \"description\": \"Get a list of top tourist attractions for a given city.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"The name of the city to find attractions for.\"}}, \"required\": [\"city\"]}}}]"
    },
    {
      "role": "user",
      "content": "How should I explain the Internet?"
    }
  ])";

  auto err = OrtxApplyChatTemplate(
    tokenizer.get(), nullptr,
    messages_json.c_str(), nullptr,
    templated_text.ToBeAssigned(), true, false);

  if (err != kOrtxOK) {
    std::cout << "Failed to apply chat template (Minja/Phi-4 tools)" << std::endl;
    std::cout << "Error code: " << err << std::endl;
    std::cout << "Error message: " << OrtxGetLastErrorMessage() << std::endl;
  }

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(templated_text.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(tensor.Code(), kOrtxOK);

  const char* text_ptr = nullptr;
  OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr);

  std::string expected_output = "<|system|>You are a helpful assistant.<|tool|>"
                                "[{\"name\": \"get_weather\", \"description\": \"Get the current weather for a given location.\", "
                                "\"parameters\": {\"location\": {\"type\": \"str\", \"description\": \"The name of the city or location.\"}}}, "
                                "{\"name\": \"get_tourist_attractions\", \"description\": \"Get a list of top tourist attractions for a given city.\", "
                                "\"parameters\": {\"city\": {\"type\": \"str\", \"description\": \"The name of the city to find attractions for.\"}}}]"
                                "<|/tool|><|end|><|user|>How should I explain the Internet?<|end|><|assistant|>";

  ASSERT_EQ(std::string(text_ptr), expected_output);
}

TEST(OrtxTokenizerTest, Qwen2_5_ChatTemplateWithMinjaTools) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/qwen2.5");
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

  // Minja/Phi-4 style tools
  std::string tools_json = R"(
    [
      {
        "name": "get_horoscope",
        "description": "Get today's horoscope for an astrological sign.",
        "parameters": {
          "sign": {
            "type": "str",
            "description": "An astrological sign like Taurus or Aquarius",
            "default": ""
          }
        }
      }
    ])";

  auto err = OrtxApplyChatTemplate(
    tokenizer.get(), nullptr,
    messages_json.c_str(), tools_json.c_str(),
    templated_text.ToBeAssigned(), true, false);

  if (err != kOrtxOK) {
    std::cout << "Failed to apply chat template (Minja tools)" << std::endl;
    std::cout << "Error code: " << err << std::endl;
    std::cout << "Error message: " << OrtxGetLastErrorMessage() << std::endl;
  }

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(templated_text.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(tensor.Code(), kOrtxOK);

  const char* text_ptr = nullptr;
  OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr);

    std::string expected_output = "<|im_start|>system\n"
                                  "You are a helpful assistant.\n\n"
                                  "# Tools\n\n"
                                  "You may call one or more functions to assist with the user query.\n\n"
                                  "You are provided with function signatures within <tools></tools> XML tags:\n"
                                  "<tools>\n"
                                  "{\"name\": \"get_horoscope\", \"description\": \"Get today's horoscope for an astrological sign.\", \"parameters\": {\"sign\": {\"type\": \"str\", \"description\": \"An astrological sign like Taurus or Aquarius\", \"default\": \"\"}}}\n"
                                  "</tools>\n\n"
                                  "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                                  "<tool_call>\n"
                                  "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                                  "</tool_call><|im_end|>\n"
                                  "<|im_start|>user\n"
                                  "How should I explain the Internet?<|im_end|>\n"
                                  "<|im_start|>assistant\n";

  ASSERT_EQ(std::string(text_ptr), expected_output);
}

TEST(OrtxTokenizerTest, Qwen2_5_ChatTemplateWithOAIToolType) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/qwen2.5");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK);

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

  // OpenAI "type": "tool" format
  std::string tools_json = R"(
    [
      {
        "type": "tool",
        "name": "get_horoscope",
        "description": "Get today's horoscope for an astrological sign.",
        "parameters": {
          "type": "object",
          "properties": {
            "sign": {
              "type": "string",
              "description": "An astrological sign like Taurus or Aquarius"
            }
          },
          "required": ["sign"]
        }
      }
    ])";

  auto err = OrtxApplyChatTemplate(
    tokenizer.get(), nullptr,
    messages_json.c_str(), tools_json.c_str(),
    templated_text.ToBeAssigned(), true, false);

  if (err != kOrtxOK) {
    std::cout << "Failed to apply chat template (OAI tool type)" << std::endl;
    std::cout << "Error code: " << err << std::endl;
    std::cout << "Error message: " << OrtxGetLastErrorMessage() << std::endl;
  }

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(templated_text.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(tensor.Code(), kOrtxOK);

  const char* text_ptr = nullptr;
  OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr);

  std::string expected_output = "<|im_start|>system\n"
                                "You are a helpful assistant.\n\n"
                                "# Tools\n\n"
                                "You may call one or more functions to assist with the user query.\n\n"
                                "You are provided with function signatures within <tools></tools> XML tags:\n"
                                "<tools>\n"
                                "{\"name\": \"get_horoscope\", \"description\": \"Get today's horoscope for an astrological sign.\", \"parameters\": {\"sign\": {\"type\": \"str\", \"description\": \"An astrological sign like Taurus or Aquarius\"}}}\n"
                                "</tools>\n\n"
                                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                                "<tool_call>\n"
                                "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                                "</tool_call><|im_end|>\n"
                                "<|im_start|>user\n"
                                "How should I explain the Internet?<|im_end|>\n"
                                "<|im_start|>assistant\n";

  ASSERT_EQ(std::string(text_ptr), expected_output);
}

TEST(OrtxTokenizerTest, Qwen2_5_ChatTemplateWithOAIFunctionType) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/qwen2.5");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK);

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

  // OpenAI "type": "function" format
  std::string tools_json = R"(
    [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a given location.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The name of the city or location."
              }
            },
            "required": ["location"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "get_tourist_attractions",
          "description": "Get a list of top tourist attractions for a given city.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string",
                "description": "The name of the city to find attractions for."
              }
            },
            "required": ["city"]
          }
        }
      }
    ])";

  auto err = OrtxApplyChatTemplate(
    tokenizer.get(), nullptr,
    messages_json.c_str(), tools_json.c_str(),
    templated_text.ToBeAssigned(), true, false);

  if (err != kOrtxOK) {
    std::cout << "Failed to apply chat template (OAI function type)" << std::endl;
    std::cout << "Error code: " << err << std::endl;
    std::cout << "Error message: " << OrtxGetLastErrorMessage() << std::endl;
  }

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(templated_text.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(tensor.Code(), kOrtxOK);

  const char* text_ptr = nullptr;
  OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&text_ptr), nullptr, nullptr);

  std::string expected_output = "<|im_start|>system\n"
                                "You are a helpful assistant.\n\n"
                                "# Tools\n\n"
                                "You may call one or more functions to assist with the user query.\n\n"
                                "You are provided with function signatures within <tools></tools> XML tags:\n"
                                "<tools>\n"
                                "{\"name\": \"get_weather\", \"description\": \"Get the current weather for a given location.\", \"parameters\": {\"location\": {\"type\": \"str\", \"description\": \"The name of the city or location.\"}}}\n"
                                "{\"name\": \"get_tourist_attractions\", \"description\": \"Get a list of top tourist attractions for a given city.\", \"parameters\": {\"city\": {\"type\": \"str\", \"description\": \"The name of the city to find attractions for.\"}}}\n"
                                "</tools>\n\n"
                                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                                "<tool_call>\n"
                                "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                                "</tool_call><|im_end|>\n"
                                "<|im_start|>user\n"
                                "How should I explain the Internet?<|im_end|>\n"
                                "<|im_start|>assistant\n";

  ASSERT_EQ(std::string(text_ptr), expected_output);
}
