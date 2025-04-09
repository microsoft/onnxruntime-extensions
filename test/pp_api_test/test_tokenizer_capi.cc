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

TEST(CApiTest, ApiTest) {
  int ver = OrtxGetAPIVersion();
  EXPECT_GT(ver, 0);
  OrtxTokenizer* tokenizer = NULL;
  extError_t err = OrtxCreateTokenizer(&tokenizer, "data/llama2");
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
    EXPECT_EQ(err, kOrtxOK);
#ifdef _DEBUG
    std::cout << token;
#endif
  }

#ifdef _DEBUG
  std::cout << std::endl;
#endif

  OrtxDisposeOnly(detok_cache);
  OrtxDispose(&tokenizer);
}

TEST(OrtxTokenizerTest, WhisperTokenizer) {
  // test the llama2 tokenizer with BPE class, instead of sentencepiece wrapper.
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/tokenizer/whisper.tiny");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  OrtxObjectPtr<OrtxTokenId2DArray> prompt_ids;

  extError_t err = OrtxGetDecoderPromptIds(tokenizer.get(), 1, "en", "transcribe", 1, prompt_ids.ToBeAssigned());
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

TEST(OrtxTokenizerTest, SpmUgmTokenizer) {
  // test the llama2 tokenizer with BPE class, instead of sentencepiece wrapper.
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/tokenizer/fairseq/xlm-roberta-base");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  const char* input[] = {"I like walking my cute dog\n and\x17 then, 生活的真谛是  \t\t\t\t \n\n61"};
  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  EXPECT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  // expected ids was generated using the following command:
  // AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({
    0, 87, 1884, 122395, 759, 99942, 10269, 136, 7068, 4, 6, 62668, 5364, 245875, 354, 11716, 2}));

  OrtxObjectPtr<OrtxStringArray> decoded_text;
  OrtxDetokenize(tokenizer.get(), token_ids.get(), decoded_text.ToBeAssigned());
  EXPECT_EQ(decoded_text.Code(), kOrtxOK);

  const char* text = nullptr;
  OrtxStringArrayGetItem(decoded_text.get(), 0, &text);
  // because the tokenization remove the character from the string, the decoded text is not the same as the input text.
  std::string filtered_text(input[0]);
  filtered_text.erase(
      std::remove_if(filtered_text.begin(), filtered_text.end(), [](unsigned char chr) { return chr < 0x20; }),
      filtered_text.end());
  // remove the consecutive spaces
  filtered_text.erase(std::unique(filtered_text.begin(), filtered_text.end(),
                                  [](char lhs, char rhs) { return lhs == ' ' && rhs == ' '; }),
                      filtered_text.end());

  EXPECT_STREQ(filtered_text.c_str(), text);
}

static std::string ReadFile(const std::string& filepath) {
  std::ifstream file(filepath.data(), std::ios::binary);
  if (!file.is_open()) {
    return "";
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

TEST(OrtxTokenizerTest, Phi3_Small_Tokenizer_Blob) {
  std::string config_blob = ReadFile("data/tokenizer/phi-3-small/tokenizer_config.json");
  ASSERT_FALSE(config_blob.empty()) << "Failed to read config blob file, stopping the test.";

  std::string raw_model_blob = ReadFile("data/tokenizer/phi-3-small/cl100k_base.tiktoken");
  ASSERT_FALSE(raw_model_blob.empty()) << "Failed to read raw model blob file, stopping the test.";

  std::string module_blob = ReadFile("data/tokenizer/phi-3-small/tokenizer_module.json");
  ASSERT_FALSE(module_blob.empty()) << "Failed to read module blob file, stopping the test.";

  struct OrtxTokenizerBlob blobs(config_blob, "", module_blob, raw_model_blob);

  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizerFromBlob, &blobs);
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  // validate tokenizer is not null
  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  std::vector<extTokenId_t> EXPECTED_IDS_0 = {2028, 374, 264, 1296, 13};
  const char* input[] = {"This is a test.",
                         "the second one",
                         "I like walking my cute dog\n and\x17 then",
                         // "Hey<|endoftext|>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61"};
                         "I like walking my cute dog\n and\x17 then 生活的真谛是 \t\t\t\t \n\n61" };


  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  OrtxTokenize(tokenizer.get(), input, 4, token_ids.ToBeAssigned());
  EXPECT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);
  EXPECT_EQ(ids_vec, EXPECTED_IDS_0);
}

TEST(OrtxTokenizerTest, Phi3TokenizerBlob) {
  std::string config_blob = ReadFile("data/phi-3/tokenizer_config.json");
  ASSERT_FALSE(config_blob.empty()) << "Failed to read config blob file, stopping the test.";

  std::string vocab_blob = ReadFile("data/phi-3/tokenizer.json");
  ASSERT_FALSE(vocab_blob.empty()) << "Failed to read vocab blob file, stopping the test.";

  struct OrtxTokenizerBlob blob(config_blob, vocab_blob, "", "");

  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizerFromBlob, &blob);
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  // validate tokenizer is not null
  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  const char* input[] = {"I like walking my cute dog\n and\x17 then, 生活的真谛是  \t\t\t\t \n\n61"};
  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  EXPECT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  // expected ids was generated using the following command:
  // AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
  EXPECT_EQ(ids_vec,
            std::vector<extTokenId_t>({1,   306,   763,   22049, 590,   274,   1082,  11203, 13,    322,  26,
                                       769, 29892, 29871, 30486, 31704, 30210, 30848, 235,   179,   158,  30392,
                                       259, 12,    12,    12,    12,    29871, 13,    13,    29953, 29896}));
}

TEST(OrtxTokenizerTest, T5Tokenizer) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/tokenizer/t5-small");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  const char* input[] = {"I <extra_id_0> like walking my cute dog\n and\x17 then, 生活的真谛是  \t\t\t\t \n\n61"};
  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  ASSERT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  // AutoTokenizer.from_pretrained("google-t5/t5-small")
  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({
    27, 3, 32099, 114, 3214, 82, 5295, 1782, 11, 258, 6, 3, 2, 3, 4241, 1}));
}

TEST(OrtxTokenizerTest, ChatGLMTokenizer) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/tokenizer/THUDM/chatglm-6b");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  const char* input[] = {"I like walking my cute dog\n and\x17 then, 生活的真谛是  \t\t\t\t \n\n61"};
  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  ASSERT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  // AutoTokenizer.from_pretrained("data/tokenizer/THUDM/chatglm-6b", trust_remote_code=True)
  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({
    115, 176, 3867, 162, 9251, 2829, 4, 102, 220, 6, 5, 63977, 91446,
    63829, 130009, 130008, 130008, 130008, 130008, 5, 4, 4, 21, 9, 130001, 130004}));
}

TEST(OrtxTokenizerTest, MarianTokenizer) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/tokenizer/nmt");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  const char* input[] = {"Hello-there THIS Is a Test"};
  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  OrtxTokenize(tokenizer.get(), input, 1, token_ids.ToBeAssigned());
  ASSERT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  // AutoTokenizer.from_pretrained("data/tokenizer/nmt")(...)
  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({
    281, 13919, 296, 404, 352, 346, 479, 292, 9428, 0}));
}