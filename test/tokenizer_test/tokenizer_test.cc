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

// ************ NEEDS UPDATES, FIXING AND TESTING FOR ALL OTHER TESTS ************

TEST(CApiTest, StreamApiTest) {
  OrtxTokenizer* tokenizer = NULL;
  extError_t err = OrtxCreateTokenizer(&tokenizer, "data/llama2");
  EXPECT_EQ(err, kOrtxOK);
  
//   char* decoded_text = NULL;
//   err = TfmCreate(kTfmKindDetokenizerCache, &detok_cache);
//   EXPECT_EQ(err, kOrtxOK);

//   tfmTokenId_t token_ids[] = {1, 910, 338, 263, 1243, 322, 278, 1473, 697, 29889, 29871, 35};
//   for (size_t i = 0; i < sizeof(token_ids) / sizeof(token_ids[0]); i++) {
//     const char* token = NULL;
//     err = TfmDetokenizeCached(tokenizer, detok_cache, token_ids[i], &token);
// #ifdef _DEBUG
//     std::cout << token;
// #endif
//     EXPECT_EQ(err, kOrtxOK);
//   }

// #ifdef _DEBUG
//   std::cout << std::endl;
// #endif

  // free(&detok_cache);
  // free(&tokenizer);

  const char* input = "This is a test";
  char* decoded_text = NULL;
  err = tokenize_text(tokenizer, input, &decoded_text);
  EXPECT_EQ(err, kOrtxOK);
  EXPECT_STREQ(decoded_text, input);
  free(decoded_text);
}

TEST(TfmTokTest, ClipTokenizer) {
  // TfmStatus status;
  OrtxTokenizer* tokenizer = NULL;
  extError_t err = OrtxCreateTokenizer(&tokenizer, "data/clip");
  // if (!status.ok()) {
  //   std::cout << status.ToString() << std::endl;
  // }

  // validate tokenizer is not null
  EXPECT_NE(tokenizer, nullptr);

  //std::vector<std::string_view> input = {"this is a test", "the second one"};
  const char* input = "This is a test";
  
  char* decoded_text = NULL;
  err = tokenize_text(tokenizer, input, &decoded_text);
  EXPECT_EQ(err, kOrtxOK);
  EXPECT_STREQ(decoded_text, input);
  free(decoded_text);
  // EXPECT_TRUE(tokenization_result.ok());
  // EXPECT_EQ(token_ids.size(), 2);
  // EXPECT_EQ(token_ids[0].size(), 6);
  // EXPECT_EQ(token_ids[1].size(), 5);

  // std::vector<std::string> out_text;
  // std::vector<tfm::span<tfmTokenId_t const>> token_ids_span = {token_ids[0], token_ids[1]};
  // auto result = tokenizer->Detokenize(token_ids_span, out_text);
  // EXPECT_TRUE(result.ok());
  // EXPECT_EQ(out_text[0], input[0]);
}

TEST(TfmTokTest, GemmaTokenizer) {
  // TfmStatus status;
  OrtxTokenizer* tokenizer = NULL;
  extError_t err = OrtxCreateTokenizer(&tokenizer, "data/gemma");
  // if (!status.ok()) {
  //   std::cout << status.ToString() << std::endl;
  // }
  
  EXPECT_NE(tokenizer, nullptr);

  const char* input = "This is a test";
  
  char* decoded_text = NULL;
  err = tokenize_text(tokenizer, input, &decoded_text);
  EXPECT_EQ(err, kOrtxOK);
  EXPECT_STREQ(decoded_text, input);
  free(decoded_text);

  // std::vector<std::string_view> input = {
  //     "I like walking my cute dog\n and\x17 then",
  //     "生活的真谛是",
  //     "\t\t\t\t \n\n61",
  //     "Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61"};
  // std::vector<tfmTokenId_t> EXPECTED_IDS_0 = {2, 235285, 1154, 10350, 970, 9786, 5929, 108, 578, 240, 1492};
  // std::vector<tfmTokenId_t> EXPECTED_IDS_1 = {2, 122182, 235710, 245467, 235427};
  // std::vector<tfmTokenId_t> EXPECTED_IDS_2 = {2, 255971, 235248, 109, 235318, 235274};
  // std::vector<tfmTokenId_t> EXPECTED_IDS_3 = {2, 6750, 1, 235265, 235248, 255969, 235248, 109, 4747, 139, 235335, 139,
  //                                             216311, 241316, 139, 239880, 235341, 144, 235269, 235248, 235274, 235284,
  //                                             235304, 235310, 235248, 235274, 235308, 235248, 235308, 235269, 235318, 235274};

  // std::vector<std::vector<tfmTokenId_t>> token_ids;
  // auto tokenization_result = tokenizer->Tokenize(input, token_ids);
  // EXPECT_TRUE(tokenization_result.ok());
  // EXPECT_EQ(token_ids.size(), input.size());
  // DumpTokenIds(token_ids);
  // EXPECT_EQ(token_ids[0], EXPECTED_IDS_0);
  // EXPECT_EQ(token_ids[1], EXPECTED_IDS_1);
  // EXPECT_EQ(token_ids[2], EXPECTED_IDS_2);
  // EXPECT_EQ(token_ids[3], EXPECTED_IDS_3);

  // std::vector<std::string> out_text;
  // std::vector<tfm::span<tfmTokenId_t const>> token_ids_span = {
  //   EXPECTED_IDS_0, EXPECTED_IDS_1, EXPECTED_IDS_2, EXPECTED_IDS_3};
  // auto result = tokenizer->Detokenize(token_ids_span, out_text);
  // EXPECT_TRUE(result.ok());
  // // std::cout << out_text[0] << std::endl;
  // // std::cout << out_text[1] << std::endl;
  // // std::cout << out_text[2] << std::endl;
  // EXPECT_EQ(out_text[0], input[0]);
  // EXPECT_EQ(out_text[1], input[1]);
}

static const char* kPromptText = R"(```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """
   primes = []
   for num in range(2, n+1):
       is_prime = True
       for i in range(2, int(math.sqrt(num))+1):
           if num % i == 0:
               is_prime = False
               break
       if is_prime:
           primes.append(num)
   print(primes)''')";

TEST(TfmTokTest, CodeGenTokenizer) {
  // TfmStatus status;
  OrtxTokenizer* tokenizer = NULL;
  extError_t err = OrtxCreateTokenizer(&tokenizer, "data/phi-2");
  // if (!status.ok()) {
  //   std::cout << status.ToString() << std::endl;
  // }

  // validate tokenizer is not null
  EXPECT_NE(tokenizer, nullptr);

  const char* prompt_text = kPromptText;
  
  char* decoded_text = NULL;
  err = tokenize_text(tokenizer, prompt_text, &decoded_text);
  EXPECT_EQ(err, kOrtxOK);
  EXPECT_STREQ(decoded_text, prompt_text);
  free(decoded_text);

  // std::vector<std::string_view> input = {prompt_text};
  // std::vector<std::vector<tfmTokenId_t>> token_ids;
  // auto tokenization_result = tokenizer->Tokenize(input, token_ids);
  // EXPECT_TRUE(tokenization_result.ok());
  // EXPECT_EQ(token_ids.size(), 1);

  // std::vector<std::string> out_text;
  // std::vector<tfm::span<tfmTokenId_t const>> token_ids_span = {token_ids[0]};
  // auto result = tokenizer->Detokenize(token_ids_span, out_text);
  // EXPECT_TRUE(result.ok());
  // //  std::cout << out_text[0] << std::endl;
  // EXPECT_EQ(out_text[0], input[0]);
}

TEST(TfmTokStreamTest, CodeGenTokenizer) {
  // TfmStatus status;
  OrtxTokenizer* tokenizer = NULL;
  extError_t err = OrtxCreateTokenizer(&tokenizer, "data/phi-2");
  // if (!status.ok()) {
  //   std::cout << status.ToString() << std::endl;
  // }

  // validate tokenizer is not null
  EXPECT_NE(tokenizer, nullptr);

  const char* prompt_text = kPromptText;

  char* decoded_text = NULL;
  err = tokenize_text(tokenizer, prompt_text, &decoded_text);
  EXPECT_EQ(err, kOrtxOK);
  EXPECT_STREQ(decoded_text, prompt_text);
  free(decoded_text);

  // std::vector<std::string_view> input = {prompt_text};
  // std::vector<std::vector<tfmTokenId_t>> token_ids;
  // auto tokenization_result = tokenizer->Tokenize(input, token_ids);
  // EXPECT_TRUE(tokenization_result.ok());
  // EXPECT_EQ(token_ids.size(), 1);

  // std::string text;
  // std::unique_ptr<tfm::DecoderState> decoder_cache;
  // // token_ids[0].insert(token_ids[0].begin() + 2, 607);  // <0x20>
  // token_ids[0] = {921, 765, 2130, 588, 262, 6123, 447, 251, 2130, 588, 262};
  // for (const auto& token_id : token_ids[0]) {
  //   std::string token;
  //   status = tokenizer->Id2Token(token_id, token, decoder_cache);
  //   EXPECT_TRUE(status.ok());
  //   // std::cout << token;
  //   text.append(token);
  // }

  // EXPECT_EQ(text, input[0]);
}

TEST(TfmTokStreamTest, Llama2Tokenizer) {
  // test the llama2 tokenizer with BPE class, instead of sentencepiece wrapper.
  OrtxTokenizer* tokenizer = NULL;
  extError_t err = OrtxCreateTokenizer(&tokenizer, "data/llama2");

  // validate tokenizer is not null
  EXPECT_TRUE(tokenizer != nullptr);

  const char* input = "This is a test";
  
  char* decoded_text = NULL;
  err = tokenize_text(tokenizer, input, &decoded_text);
  EXPECT_EQ(err, kOrtxOK);
  EXPECT_STREQ(decoded_text, input);
  free(decoded_text);

  // std::vector<std::string_view> input = {"This is a test and the second one. "};
  // std::vector<std::vector<tfmTokenId_t>> token_ids;
  // auto tokenization_result = tokenizer->Tokenize(input, token_ids);
  // EXPECT_TRUE(tokenization_result.ok());
  // // Add an extra byte token for decoding tests
  // token_ids[0].push_back(35);  // <0x20>
  // DumpTokenIds(token_ids);

  // std::string text;
  // std::unique_ptr<tfm::DecoderState> decoder_cache;
  // // std::cout << "\"";
  // for (const auto& token_id : token_ids[0]) {
  //   std::string token;
  //   auto status = tokenizer->Id2Token(token_id, token, decoder_cache);
  //   EXPECT_TRUE(status.ok());
  //   // std::cout << token;
  //   text.append(token);
  // }

  // // std::cout << "\"" << std::endl;
  // EXPECT_EQ(std::string(text), std::string(input[0])); /* + " ");  // from the extra byte token */
}