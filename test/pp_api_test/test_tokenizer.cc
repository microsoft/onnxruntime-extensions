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

TEST(OrtxTokenizerTest, RegexTest) {
  std::u32string str = U"CAN'T \r\n 2413m";
  auto regcmp = std::make_unique<ort_extensions::bpe::TokenWithRegularExp>();

  std::vector<std::u32string> res;
  std::vector<std::u32string> out_tokens = {U"CAN", U"'T", U" \r\n", U" ", U"241", U"3", U"m"};

  int64_t max_length = out_tokens.size();
  regcmp->Set(str.c_str());
  std::string regex_expr = regcmp->LLAMA_REGEX_PATTERN;

  while (static_cast<int64_t>(res.size()) < max_length) {
    auto [b, tok] = regcmp->GetNextToken(regex_expr);
    res.push_back(ustring(tok));
  }
  EXPECT_EQ(res, out_tokens);
}

TEST(OrtxTokenizerTest, RegexMatchSTDTest) {
  std::vector<std::string> regex_expressions = {"'s|'t|'re|'ve|'m|'ll|'d",
                                                "\\s+",
                                                "[A-Za-z]+"};

  std::vector<std::u32string> input_strings = {U"not its, or IT'S, but it's",
                                               U"   ",
                                               U"AbCd"};                      
  auto regcmp = std::make_unique<ort_extensions::bpe::TokenWithRegularExp>();

  std::vector<std::vector<std::u32string>> res_vector;
  std::vector<std::vector<std::u32string>> out_tokens = {{U"'s"},
                                                         {U"   "},
                                                         {U"AbCd"}};

  for (auto i = 0; i < regex_expressions.size(); i++){
    int64_t max_length = out_tokens[i].size();
    regcmp->Set(input_strings[i].c_str());
    std::string regex_expr = regex_expressions[i];
    std::vector<std::u32string> res;

    while (static_cast<int64_t>(res.size()) < max_length) {
      res.push_back(regcmp->RegexMatchSTD(ustring(regex_expr)));
    }

    res_vector.push_back(res);
  }
  EXPECT_EQ(res_vector, out_tokens);
}

TEST(OrtxTokenizerTest, WrapStandaloneCategoriesTest) {
  std::vector<std::string> regex_expressions = {"[^\\p{rn}\\p{L}\\p{N}]?\\p{L}+",
                                                "\\p{rn}\\p{L}\\p{N}\\p{L}",
                                                "\\p{Z}*[\\p{rn}]+",
                                                "\\p{Z}+"};
  auto regcmp = std::make_unique<ort_extensions::bpe::TokenWithRegularExp>();

  std::vector<std::string> res;
  std::vector<std::string> out_regex = {"[^\\p{rn}\\p{L}\\p{N}]?[\\p{L}]+",
                                        "[\\p{rn}][\\p{L}][\\p{N}][\\p{L}]",
                                        "[\\p{Z}]*[\\p{rn}]+",
                                        "[\\p{Z}]+"};

  for (auto regex : regex_expressions){
    res.push_back(regcmp->WrapStandaloneCategories(regex));
  }
  EXPECT_EQ(res, out_regex);
}

TEST(OrtxTokenizerTest, RegexMatchGeneralTest) {
  std::vector<std::string> regex_expressions = {"[^\\p{rn}\\p{L}\\p{N}]?\\p{L}+",
                                                "\\p{N}{1,3}",
                                                "\\p{N}{1,5}",
                                                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
                                                "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
                                                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
                                                "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
                                                "\\p{N}{1,3}|?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+",
                                                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
                                                "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
                                                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
                                                "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
                                                "\\p{N}{1,3}|?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+"};

  std::vector<std::u32string> input_strings = {U"CAN'T \r\n ",
                                               U"2413m",
                                               U"241356m",
                                               U"Ich liebe München <3 \r\n ",
                                               U"生活的真谛是"};                      
  auto regcmp = std::make_unique<ort_extensions::bpe::TokenWithRegularExp>();

  std::vector<std::vector<std::u32string>> res_vector;
  std::vector<std::vector<std::u32string>> out_tokens = {{U"CAN", U"'T", U"", U""},
                                                         {U"241", U"3"},
                                                         {U"24135", U"6"},
                                                         {U"Ich", U" liebe", U" München", U" <", U"3", U" \r\n", U" "},
                                                         {U"生活的真谛是"}};

  for (auto i = 0; i < regex_expressions.size(); i++){
    int64_t max_length = out_tokens[i].size();
    regcmp->Set(input_strings[i].c_str());
    std::string regex_expr = regex_expressions[i];
    std::vector<std::u32string> res;

    while (static_cast<int64_t>(res.size()) < max_length) {
      res.push_back(regcmp->RegexMatchGeneral(regex_expr));
    }

    res_vector.push_back(res);
  }
  EXPECT_EQ(res_vector, out_tokens);
}

TEST(OrtxTokenizerTest, ClipTokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/tokenizer/clip");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  // validate tokenizer is not null
    ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

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
    tokenizer.reset();
  }

  // validate tokenizer is not null
    ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

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

TEST(OrtxTokenizerTest, Phi3_Small_Hf_Tokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/tokenizer/phi-3-small-cvt");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  // validate tokenizer is not null
    ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  std::vector<extTokenId_t> EXPECTED_IDS_0 = {2028, 374, 264, 1296, 13};
  std::vector<std::string_view> input = {"This is a test.", "Ich liebe München",
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
  auto status = tokenizer->Load("data/tokenizer/phi-3-small");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  // validate tokenizer is not null
    ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

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
    tokenizer.reset();
  }

  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

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
    tokenizer.reset();
  }

  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

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
    tokenizer.reset();
  }

  // validate tokenizer is not null
  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

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

  // 252 and the following ids cannot be decoded as a valid utf-8 string
  std::vector<extTokenId_t> invalid_token_ids_span = {14675, 8466, 705, 252, 538, 5374, 82, 329, 4554};
  std::vector<std::string> out_text1;
  status = tokenizer->Detokenize({ort_extensions::span<const extTokenId_t>(invalid_token_ids_span)}, out_text1);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(out_text1.size(), 1);
  std::string out_text_ref = out_text1.back();
  // std::cout << out_text_ref << std::endl;
  EXPECT_EQ(out_text_ref.substr(out_text_ref.length() - 3, 3), "\ufffd");
}

TEST(OrtxTokenizerStreamTest, CodeGenTokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/phi-2");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  // validate tokenizer is not null
  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  const char* prompt_text = kPromptText;

  std::vector<std::string_view> input = {prompt_text};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(token_ids.size(), 1);

  std::string text;
  std::unique_ptr<ort_extensions::TokenizerDecodingState> decoder_cache;
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
    tokenizer.reset();
  }

  // validate tokenizer is not null
  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  std::vector<std::string_view> input = {"This is a test and the second one is in German. Ich liebe München!"};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  // Add an extra byte token for decoding tests
  token_ids[0].push_back(35);  // <0x20>
  DumpTokenIds(token_ids);

  std::string text;
  std::unique_ptr<ort_extensions::TokenizerDecodingState> decoder_cache;
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
    tokenizer.reset();
  }

  // validate tokenizer is not null
  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  std::vector<std::string_view> input = {
      R"(こんにちは。データ分析にはいくつかのステップがあります。まずは目的を明確にします。次に、データを収集し、クリーニングを行います。)"
      R"(その後、データを構造化し、その後、データを分析します。これらのステップを実行することで、データを有意的に分析することができます。)"};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  // Add an extra byte token for decoding tests
  token_ids[0].push_back(35);  // <0x20>
  DumpTokenIds(token_ids);

  std::string text;
  std::unique_ptr<ort_extensions::TokenizerDecodingState> decoder_cache;
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
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

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

TEST(OrtxTokenizerTest, SpmUgmTokenizer) {
  // test the llama2 tokenizer with BPE class, instead of sentencepiece wrapper.
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/tokenizer/fairseq/xlm-roberta-base");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << "Failed to create tokenizer, stopping the test.";

  const char* input[] = {"I like walking my cute dog\n and\x17 then, 生活的真谛是  \t\t\t\t \n\n61"};
  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  OrtxTokenize(tokenizer.get(), input, 1, ort_extensions::ptr(token_ids));
  EXPECT_EQ(token_ids.Code(), kOrtxOK);

  size_t length = 0;
  const extTokenId_t* ids = nullptr;
  OrtxTokenId2DArrayGetItem(token_ids.get(), 0, &ids, &length);
  std::vector<extTokenId_t> ids_vec(ids, ids + length);

  // expected ids was generated using the following command:
  // AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
  EXPECT_EQ(ids_vec, std::vector<extTokenId_t>({0, 87, 1884, 122395, 759, 99942, 10269, 136, 7068, 4, 6, 62668, 5364,
                                                245875, 354, 11716, 2}));

  OrtxObjectPtr<OrtxStringArray> decoded_text;
  OrtxDetokenize(tokenizer.get(), token_ids.get(), ort_extensions::ptr(decoded_text));
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
};

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
                         "Hey<|endoftext|>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61"};

  OrtxObjectPtr<OrtxTokenId2DArray> token_ids;
  OrtxTokenize(tokenizer.get(), input, 4, ort_extensions::ptr(token_ids));
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
  OrtxTokenize(tokenizer.get(), input, 1, ort_extensions::ptr(token_ids));
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
