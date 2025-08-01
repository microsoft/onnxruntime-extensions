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

/*

Similar to the Phi4MiniReasoningChatTemplate test, this test checks for chat template loading
from a separate chat_template.json file, instead of a key in tokenizer_config.json. Although
the official HuggingFace model files for the gemma-3 model contain both the chat_template.json
file as well as the key in tokenizer_config.json, we have removed the key in "data/gemma-3-chat"
to test the special loading from the chat_template.json file.

*/

TEST(OrtxTokenizerTest, Gemma3SpecialChatTemplate) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/gemma-3-chat");
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
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/tokenizer/whisper.tiny");
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
  OrtxTokenizeWithOptions(tokenizer.get(), input, 1, token_ids.ToBeAssigned(), false);
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
}
