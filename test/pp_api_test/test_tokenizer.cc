// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <locale>
#include "gtest/gtest.h"

#include "c_only_test.h"
#include "ortx_cpp_helper.h"
#include "shared/api/tokenizer_impl.h"

static void DumpTokenIds(const std::vector<std::vector<extTokenId_t>>& token_ids) {
#ifdef _DEBUG
  for (const auto& tokens : token_ids) {
    for (const auto& token : tokens) {
      std::cout << token << " ";
    }

    std::cout << std::endl;
  }

  std::cout << std::endl;
#endif
}

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

TEST(CApiTest, StreamApiTest) {
  OrtxTokenizer* tokenizer = NULL;
  extError_t err = OrtxCreate(kOrtxKindTokenizer, &tokenizer, "data/llama2");
  EXPECT_EQ(err, kOrtxOK);

  OrtxDetokenizerCache* detok_cache = NULL;
  err = OrtxCreate(kOrtxKindDetokenizerCache, &detok_cache);
  EXPECT_EQ(err, kOrtxOK);

  extTokenId_t token_ids[] = {1, 910, 338, 263, 1243, 322, 278, 1473, 697, 29889, 29871, 35};
  for (size_t i = 0; i < sizeof(token_ids) / sizeof(token_ids[0]); i++) {
    const char* token = NULL;
    err = OrtxDetokenizeCached(tokenizer, detok_cache, token_ids[i], &token);
#ifdef _DEBUG
    std::cout << token;
#endif
    EXPECT_EQ(err, kOrtxOK);
  }

#ifdef _DEBUG
  std::cout << std::endl;
#endif

  OrtxDisposeOnly(detok_cache);
  OrtxDispose(&tokenizer);
}

TEST(OrtxTokenizerTest, ClipTokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/clip");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
  }

  // validate tokenizer is not null
  EXPECT_NE(tokenizer, nullptr);

  std::vector<std::string_view> input = {"this is a test", "the second one"};

  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(token_ids.size(), 2);
  EXPECT_EQ(token_ids[0].size(), 6);
  EXPECT_EQ(token_ids[1].size(), 5);

  std::vector<std::string> out_text;
  std::vector<ort_extensions::span<extTokenId_t const>> token_ids_span = {token_ids[0], token_ids[1]};
  status = tokenizer->Detokenize(token_ids_span, out_text);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(out_text[0], input[0]);
}

TEST(OrtxTokenizerTest, TicTokenTokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/tiktoken");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
  }

  // validate tokenizer is not null
  EXPECT_NE(tokenizer, nullptr);

  std::vector<extTokenId_t> EXPECTED_IDS_0 = {128000, 2028, 374, 264, 1296};
  std::vector<std::string_view> input = {"This is a test", "the second one"};

  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(token_ids.size(), 2);
  EXPECT_EQ(token_ids[0], EXPECTED_IDS_0);
  EXPECT_EQ(token_ids[1].size(), 4);

  std::vector<std::string> out_text;
  std::vector<ort_extensions::span<extTokenId_t const>> token_ids_span = {token_ids[0], token_ids[1]};
  status = tokenizer->Detokenize(token_ids_span, out_text);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(out_text[0], input[0]);
}

TEST(OrtxTokenizerTest, Phi3_S_Tokenizer) {
  if (!std::filesystem::exists("data2/phi-3-small")) {
    GTEST_SKIP() << "Skip test as extra test data is not deployed.";
  }

  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data2/phi-3-small");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
  }

  // validate tokenizer is not null
  EXPECT_NE(tokenizer, nullptr);

  std::vector<extTokenId_t> EXPECTED_IDS_0 = {2028, 374, 264, 1296, 13};
  std::vector<std::string_view> input = {"This is a test.", "the second one",
                                         "I like walking my cute dog\n and\x17 then",
                                         "Hey<|endoftext|>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61"};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  DumpTokenIds(token_ids);

  EXPECT_EQ(token_ids.size(), input.size());
  EXPECT_EQ(token_ids[0], EXPECTED_IDS_0);

  std::vector<std::string> out_text;
  std::vector<ort_extensions::span<extTokenId_t const>> token_ids_span = {token_ids[0], token_ids[1]};
  status = tokenizer->Detokenize(token_ids_span, out_text);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(out_text[0], input[0]);
}

TEST(OrtxTokenizerTest, Phi3_Small_Tokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/phi-3-small");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
  }

  // validate tokenizer is not null
  EXPECT_NE(tokenizer, nullptr);

  std::vector<extTokenId_t> EXPECTED_IDS_0 = {2028, 374, 264, 1296, 13};
  std::vector<std::string_view> input = {
      "This is a test.",
      "the second one",
      "I like walking my cute dog\n and\x17 then", 
      "Hey<|endoftext|>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61"};
  std::vector<std::vector<extTokenId_t>>
      token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  DumpTokenIds(token_ids);

  EXPECT_EQ(token_ids.size(), input.size());
  EXPECT_EQ(token_ids[0], EXPECTED_IDS_0);
}

TEST(OrtxTokenizerTest, GemmaTokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/gemma");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
  }

  std::vector<std::string_view> input = {"I like walking my cute dog\n and\x17 then", "生活的真谛是", "\t\t\t\t \n\n61",
                                         "Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61"};
  std::vector<extTokenId_t> EXPECTED_IDS_0 = {2, 235285, 1154, 10350, 970, 9786, 5929, 108, 578, 240, 1492};
  std::vector<extTokenId_t> EXPECTED_IDS_1 = {2, 122182, 235710, 245467, 235427};
  std::vector<extTokenId_t> EXPECTED_IDS_2 = {2, 255971, 235248, 109, 235318, 235274};
  std::vector<extTokenId_t> EXPECTED_IDS_3 = {2,      6750,   1,      235265, 235248, 255969, 235248, 109,
                                              4747,   139,    235335, 139,    216311, 241316, 139,    239880,
                                              235341, 144,    235269, 235248, 235274, 235284, 235304, 235310,
                                              235248, 235274, 235308, 235248, 235308, 235269, 235318, 235274};

  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(token_ids.size(), input.size());
  DumpTokenIds(token_ids);
  EXPECT_EQ(token_ids[0], EXPECTED_IDS_0);
  EXPECT_EQ(token_ids[1], EXPECTED_IDS_1);
  EXPECT_EQ(token_ids[2], EXPECTED_IDS_2);
  EXPECT_EQ(token_ids[3], EXPECTED_IDS_3);

  std::vector<std::string> out_text;
  std::vector<ort_extensions::span<extTokenId_t const>> token_ids_span = {EXPECTED_IDS_0, EXPECTED_IDS_1,
                                                                          EXPECTED_IDS_2, EXPECTED_IDS_3};
  status = tokenizer->Detokenize(token_ids_span, out_text);
  EXPECT_TRUE(status.IsOk());
  // std::cout << out_text[0] << std::endl;
  // std::cout << out_text[1] << std::endl;
  // std::cout << out_text[2] << std::endl;
  EXPECT_EQ(out_text[0], input[0]);
  EXPECT_EQ(out_text[1], input[1]);
}

TEST(OrtxTokenizerTest, Phi3Tokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/phi-3");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
  }

  std::vector<std::string_view> input = {
      "分析",
      " こんにちは",  // an extra space at the beginning
      "<|user|>こんにちは。データ分析するにはなにをすればいい？<|end|><|assistant|>"};
  std::vector<extTokenId_t> EXPECTED_IDS_0 = {1, 29871, 30748, 233, 161, 147};
  std::vector<extTokenId_t> EXPECTED_IDS_1 = {1, 259, 30589, 30389, 30353, 30644, 30449};
  std::vector<extTokenId_t> EXPECTED_IDS_2 = {
      1,     32010, 29871, 30589, 30389, 30353, 30644, 30449, 30267, 30597, 30185, 30369, 30748, 233,   161,  147,
      30427, 30332, 30353, 30449, 30371, 30353, 30396, 30427, 30553, 31254, 30298, 30298, 30882, 32007, 32001};

  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(token_ids.size(), input.size());
  DumpTokenIds(token_ids);
  EXPECT_EQ(token_ids[0], EXPECTED_IDS_0);
  EXPECT_EQ(token_ids[1], EXPECTED_IDS_1);
  EXPECT_EQ(token_ids[2], EXPECTED_IDS_2);

  std::vector<std::string> out_text;
  std::vector<ort_extensions::span<extTokenId_t const>> token_ids_span = {EXPECTED_IDS_0, EXPECTED_IDS_1};
  status = tokenizer->Detokenize(token_ids_span, out_text);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(out_text[0], input[0]);
  EXPECT_EQ(out_text[1], input[1]);
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

TEST(OrtxTokenizerTest, CodeGenTokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/phi-2");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
  }

  // validate tokenizer is not null
  EXPECT_NE(tokenizer, nullptr);

  const char* prompt_text = kPromptText;

  std::vector<std::string_view> input = {prompt_text};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(token_ids.size(), 1);

  std::vector<std::string> out_text;
  std::vector<ort_extensions::span<extTokenId_t const>> token_ids_span = {token_ids[0]};
  status = tokenizer->Detokenize(token_ids_span, out_text);
  EXPECT_TRUE(status.IsOk());
  //  std::cout << out_text[0] << std::endl;
  EXPECT_EQ(out_text[0], input[0]);
}

TEST(OrtxTokenizerStreamTest, CodeGenTokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/phi-2");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
  }

  // validate tokenizer is not null
  EXPECT_NE(tokenizer, nullptr);

  const char* prompt_text = kPromptText;

  std::vector<std::string_view> input = {prompt_text};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(token_ids.size(), 1);

  std::string text;
  std::unique_ptr<ort_extensions::BPEDecoderState> decoder_cache;
  // token_ids[0].insert(token_ids[0].begin() + 2, 607);  // <0x20>
  token_ids[0] = {564, 921, 765, 2130, 588, 262, 6123, 447, 251, 2130, 588, 262};
  for (const auto& token_id : token_ids[0]) {
    std::string token;
    status = tokenizer->Id2Token(token_id, token, decoder_cache);
    EXPECT_TRUE(status.IsOk());
    // std::cout << token;
    text.append(token);
  }

  // EXPECT_EQ(text, input[0]);
}

TEST(OrtxTokenizerStreamTest, Llama2Tokenizer) {
  // test the llama2 tokenizer with BPE class, instead of sentencepiece wrapper.
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/llama2");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
  }

  // validate tokenizer is not null
  EXPECT_TRUE(tokenizer != nullptr);

  std::vector<std::string_view> input = {"This is a test and the second one. "};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  // Add an extra byte token for decoding tests
  token_ids[0].push_back(35);  // <0x20>
  DumpTokenIds(token_ids);

  std::string text;
  std::unique_ptr<ort_extensions::BPEDecoderState> decoder_cache;
  // std::cout << "\"";
  for (const auto& token_id : token_ids[0]) {
    std::string token;
    auto status = tokenizer->Id2Token(token_id, token, decoder_cache);
    EXPECT_TRUE(status.IsOk());
    // std::cout << token;
    text.append(token);
  }

  // std::cout << "\"" << std::endl;
  EXPECT_EQ(std::string(text), std::string(input[0]) + " ");  // from the extra byte token */
}

TEST(OrtxTokenizerStreamTest, Phi3Tokenizer) {
  // test the llama2 tokenizer with BPE class, instead of sentencepiece wrapper.
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/phi-3");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
  }

  // validate tokenizer is not null
  EXPECT_TRUE(tokenizer != nullptr);

  std::vector<std::string_view> input = {
      R"(こんにちは。データ分析にはいくつかのステップがあります。まずは目的を明確にします。次に、データを収集し、クリーニングを行い ます。その後、データを構造化し、その後、データを分析します。これらのステップを実行することで、データを有意的に分析することができます。)"};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  // Add an extra byte token for decoding tests
  token_ids[0].push_back(35);  // <0x20>
  DumpTokenIds(token_ids);

  std::string text;
  std::unique_ptr<ort_extensions::BPEDecoderState> decoder_cache;
  // std::cout << "\"";
  for (const auto& token_id : token_ids[0]) {
    std::string token;
    auto status = tokenizer->Id2Token(token_id, token, decoder_cache);
    EXPECT_TRUE(status.IsOk());
    // std::cout << token;
    text.append(token);
  }

  // std::cout << "\"" << std::endl;
  EXPECT_EQ(std::string(text), std::string(input[0]) + " ");  // from the extra byte token */
}

using namespace ort_extensions;

TEST(OrtxTokenizerTest, WhisperTokenizer) {
  // test the llama2 tokenizer with BPE class, instead of sentencepiece wrapper.
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/tokenizer/whisper.tiny");
  EXPECT_EQ(tokenizer.Code(), kOrtxOK);

  OrtxObjectPtr<OrtxTokenId2DArray> prompt_ids;

  extError_t err = OrtxGetDecoderPromptIds(tokenizer.get(), 1, "en", "transcribe", 1, ort_extensions::ptr(prompt_ids));
  EXPECT_EQ(err, kOrtxOK);

  size_t length = 0;
  const extTokenId_t* token_ids = NULL;
  OrtxTokenId2DArrayGetItem(prompt_ids.get(), 0, &token_ids, &length);
  std::vector<extTokenId_t> ids(token_ids, token_ids + length);

  EXPECT_EQ(ids, std::vector<extTokenId_t>({50259, 50358, 50363}));

  extTokenId_t sot_id{};
  err = OrtxConvertTokenToId(tokenizer.get(), "<|startoftranscript|>", &sot_id);
  EXPECT_EQ(err, kOrtxOK);
  EXPECT_EQ(sot_id, 50258);
}
