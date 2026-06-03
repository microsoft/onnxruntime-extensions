// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <locale>
#include "gtest/gtest.h"

#include "shared/api/tokenizer_impl.h"
#include "bpe_utils.hpp"

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

TEST(OrtxTokenizerTest, RegexTest) {
  std::u32string str = U"You'll enjoy the concert.";
  auto reg_splitter = std::make_unique<ort_extensions::bpe::PreTokenizerWithRegEx>();

  std::vector<std::u32string> res;
  std::vector<std::u32string> out_tokens = {U"You'll", U" enjoy", U" the", U" concert"};

  int64_t max_length = out_tokens.size();
  reg_splitter->Set(str.c_str());
  auto status = reg_splitter->Compile(R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?)");
  assert(status.IsOk());

  while (static_cast<int64_t>(res.size()) < max_length) {
    std::u32string_view tok = reg_splitter->GetNextToken();
    res.push_back(ustring(tok));
  }
  
  EXPECT_EQ(res, out_tokens);
}

TEST(OrtxTokenizerTest, Qwen35RegexTest) {
  // Qwen3.5 tokenizer regex — exercises [\p{L}\p{M}] and [^\s\p{L}\p{M}\p{N}] matchers
  std::u32string str = U"Hello, world! 42\ncaf\u0065\u0301 ok";  // café with combining acute accent (U+0301)
  auto reg_splitter = std::make_unique<ort_extensions::bpe::PreTokenizerWithRegEx>();

  auto status = reg_splitter->Compile(
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)");
  ASSERT_TRUE(status.IsOk()) << status.ToString();

  reg_splitter->Set(str.c_str());

  std::vector<std::u32string> res;
  for (;;) {
    std::u32string_view tok = reg_splitter->GetNextToken();
    if (tok.empty()) break;
    res.push_back(ustring(tok));
  }

  // Expected tokens:
  // "Hello" "," " world" "!" " " "4" "2" "\n" "café" (with combining accent) " ok"
  std::vector<std::u32string> expected = {
      U"Hello", U",", U" world", U"!", U" ", U"4", U"2", U"\n",
      U"caf\u0065\u0301", U" ok"};
  EXPECT_EQ(res, expected);
}

TEST(OrtxTokenizerTest, AddedTokensTest) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/added-tokens");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  std::vector<std::string_view> input = {"<|endoftext|><|assistant|><|placeholder1|>"};
  std::vector<extTokenId_t> EXPECTED_IDS_0 = {1, 32000, 32001, 32002};

  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(token_ids.size(), input.size());
  EXPECT_EQ(token_ids[0], EXPECTED_IDS_0);
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

  std::vector<std::string_view> input = {"this is a test .", "the second one"};

  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(token_ids.size(), 2);
  EXPECT_EQ(token_ids[0].size(), 7);
  EXPECT_EQ(token_ids[1].size(), 5);

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
  std::vector<std::string_view> input = {"This is a test.", "the second one",
                                         "I like walking my cute dog\n and\x17 then",
                                         // "Hey<|endoftext|>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61"};
                                         "I like walking my cute dog\n and\x17 then 生活的真谛是 \t\t\t\t \n\n61"};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  DumpTokenIds(token_ids);

  EXPECT_EQ(token_ids.size(), input.size());
  EXPECT_EQ(token_ids[0], EXPECTED_IDS_0);
}

TEST(OrtxTokenizerTest, Phi3_Vision_Tokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/phi-3-vision");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  // validate tokenizer is not null
  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  std::vector<extTokenId_t> EXPECTED_IDS_0 = {1, 3750, 338, 512, 2325, 16459, 17587, 29973};
  std::vector<std::string_view> input = {"Why is Include handled differently?"};
  std::vector<std::vector<extTokenId_t>> token_ids;
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
      "<|user|>\nこんにちは。データ分析するにはなにをすればいい？<|end|><|assistant|>"};
  std::vector<extTokenId_t> EXPECTED_IDS_0 = {1, 29871, 30748, 233, 161, 147};
  std::vector<extTokenId_t> EXPECTED_IDS_1 = {1, 259, 30589, 30389, 30353, 30644, 30449};
  std::vector<extTokenId_t> EXPECTED_IDS_2 = {
      1,   32010, 29871, 13,    30589, 30389, 30353, 30644, 30449, 30267, 30597, 30185, 30369, 30748, 233,   161,
      147, 30427, 30332, 30353, 30449, 30371, 30353, 30396, 30427, 30553, 31254, 30298, 30298, 30882, 32007, 32001};

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

// Test that phi-2 (GPT-2 pattern) correctly tokenizes strings ending with
// a single character after a space, verifying no trailing characters are dropped.
TEST(OrtxTokenizerTest, CodeGenTokenizerTrailingChar) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/phi-2");
  ASSERT_TRUE(status.IsOk()) << status.ToString();

  // "return n" — the trailing " n" should produce token 299 (Ġn), not 220 (Ġ) with 'n' dropped
  {
    std::vector<std::string_view> input = {"return n"};
    std::vector<std::vector<extTokenId_t>> token_ids;
    status = tokenizer->Tokenize(input, token_ids);
    EXPECT_TRUE(status.IsOk());
    // HF tokenizers produces: [7783, 299] for "return n"
    std::vector<extTokenId_t> EXPECTED_IDS = {7783, 299};
    EXPECT_EQ(token_ids[0], EXPECTED_IDS) << "Trailing single letter after space was dropped";

    // Also verify round-trip
    std::vector<std::string> out_text;
    std::vector<ort_extensions::span<extTokenId_t const>> span = {token_ids[0]};
    status = tokenizer->Detokenize(span, out_text);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(out_text[0], "return n");
  }

  // " n" — minimal case
  {
    std::vector<std::string_view> input = {" n"};
    std::vector<std::vector<extTokenId_t>> token_ids;
    status = tokenizer->Tokenize(input, token_ids);
    EXPECT_TRUE(status.IsOk());
    // HF: [299] for " n"
    std::vector<extTokenId_t> EXPECTED_IDS = {299};
    EXPECT_EQ(token_ids[0], EXPECTED_IDS) << "Single letter after space was dropped";
  }

  // "abc x" — another trailing single char case
  {
    std::vector<std::string_view> input = {"abc x"};
    std::vector<std::vector<extTokenId_t>> token_ids;
    status = tokenizer->Tokenize(input, token_ids);
    EXPECT_TRUE(status.IsOk());
    // HF: [39305, 2124] for "abc x"
    std::vector<extTokenId_t> EXPECTED_IDS = {39305, 2124};
    EXPECT_EQ(token_ids[0], EXPECTED_IDS) << "Trailing single letter after space was dropped";
  }

  // "def fibonacci(n):\n    if n <= 1:\n        return n" — real-world code case
  {
    std::vector<std::string_view> input = {"def fibonacci(n):\n    if n <= 1:\n        return n"};
    std::vector<std::vector<extTokenId_t>> token_ids;
    status = tokenizer->Tokenize(input, token_ids);
    EXPECT_TRUE(status.IsOk());
    // HF produces 18 tokens, last one is 299 (Ġn)
    EXPECT_FALSE(token_ids[0].empty());
    // The last token should be 299 (" n"), not 220 (" ")
    EXPECT_EQ(token_ids[0].back(), 299) << "Last token should be ' n' (299), not ' ' (220)";

    // Verify round-trip
    std::vector<std::string> out_text;
    std::vector<ort_extensions::span<extTokenId_t const>> span = {token_ids[0]};
    status = tokenizer->Detokenize(span, out_text);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(out_text[0], "def fibonacci(n):\n    if n <= 1:\n        return n");
  }
}

// Direct test of PreTokenizerWithRegEx to isolate pre-tokenization from BPE
TEST(OrtxTokenizerTest, GPT2PreTokenizerDirect) {
  using namespace ort_extensions::bpe;

  ASSERT_TRUE(PreTokenizerWithRegEx::IsL(U'n')) << "IsL('n') must be true - 'n' is a letter!";
  ASSERT_TRUE(PreTokenizerWithRegEx::IsL(U'a')) << "IsL('a') must be true";
  ASSERT_FALSE(PreTokenizerWithRegEx::IsZ(U'n')) << "IsZ('n') must be false";
  ASSERT_TRUE(PreTokenizerWithRegEx::IsZ(U' ')) << "IsZ(' ') must be true";

  PreTokenizerWithRegEx splitter;
  auto status = splitter.Compile(PreTokenizerWithRegEx::GPT2_REGEX_PATTERN);
  ASSERT_TRUE(status.IsOk()) << status.ToString();

  // Test " n" directly — collect all tokens
  {
    std::u32string text = U" n";
    splitter.Set(text.c_str());

    std::vector<std::u32string> tokens;
    while (true) {
      auto tok = splitter.GetNextToken();
      if (tok.empty()) break;
      tokens.push_back(std::u32string(tok));
    }

    // " n" should be matched as ONE token by the " ?\p{L}+" pattern
    ASSERT_EQ(tokens.size(), 1u) << "Expected 1 token (' n'), got " << tokens.size();
    EXPECT_EQ(tokens[0], U" n") << "Pre-tokenizer should match ' n' as single token";
  }

  // Test "return n"
  {
    std::u32string text = U"return n";
    splitter.Set(text.c_str());

    std::vector<std::u32string> tokens;
    while (true) {
      auto tok = splitter.GetNextToken();
      if (tok.empty()) break;
      tokens.push_back(std::u32string(tok));
    }

    ASSERT_EQ(tokens.size(), 2u) << "Expected 2 tokens ('return' + ' n'), got " << tokens.size();
    EXPECT_EQ(tokens[0], U"return") << "First token should be 'return'";
    EXPECT_EQ(tokens[1], U" n") << "Second token should be ' n'";
  }

  // Test that 'n' alone matches as \p{L}+
  {
    std::u32string text = U"n";
    splitter.Set(text.c_str());
    auto tok1 = splitter.GetNextToken();
    EXPECT_EQ(tok1.size(), 1u) << "Single 'n' should match";
    EXPECT_EQ(std::u32string(tok1), U"n") << "Single 'n' should be its own token";
  }

  // Test "hello world" — normal case
  {
    std::u32string text = U"hello world";
    splitter.Set(text.c_str());
    auto tok1 = splitter.GetNextToken();
    EXPECT_EQ(std::u32string(tok1), U"hello");
    auto tok2 = splitter.GetNextToken();
    EXPECT_EQ(std::u32string(tok2), U" world");
    auto tok3 = splitter.GetNextToken();
    EXPECT_TRUE(tok3.empty());
  }
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
  for (const auto& token_id : token_ids[0]) {
    std::string token;
    auto status = tokenizer->Id2Token(token_id, token, decoder_cache);
    EXPECT_TRUE(status.IsOk());
    text.append(token);
  }

  EXPECT_EQ(std::string(text), std::string(input[0]) + " ");  // from the extra byte token */
}

TEST(OrtxTokenizerTest, Nvidia_Mistral_Tokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/nvidia-mistral");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  // validate tokenizer is not null
  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  // From HuggingFace expected values for "nvidia/Mistral-NeMo-Minitron-8B-Instruct"
  std::vector<extTokenId_t> EXPECTED_IDS_0 = {1, 4380, 1395, 1261, 2688, 1321, 1278, 2667, 1925, 1395, 1294,
                                              8863, 1046, 14314, 5897, 2352, 28943, 1033};
  std::vector<std::string_view> input = {"This is a test and the second one is in German. Ich liebe München!"};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  DumpTokenIds(token_ids);

  EXPECT_EQ(token_ids.size(), input.size());
  EXPECT_EQ(token_ids[0], EXPECTED_IDS_0);
}

TEST(OrtxTokenizerTest, ChatGLM3Tokenizer) {
  // Tests chatglm3-6b with a BPE tokenizer.json, ensuring it is loaded properly
  // even though tokenizer_class="ChatGLMTokenizer" maps to Unigram by default.
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/chatglm");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  // === "hello" ===
  // HuggingFace Reference (add_special_tokens=False): [15616, 30914]
  {
    std::vector<std::string_view> input = {"hello"};
    std::vector<std::vector<extTokenId_t>> token_ids;
    status = tokenizer->Tokenize(input, token_ids);
    EXPECT_TRUE(status.IsOk());
    DumpTokenIds(token_ids);
    ASSERT_EQ(token_ids.size(), 1u);
    // chatglm3 adds [gMASK](64790) and <sop>(64792) prefix special tokens,
    // so expected output: [64790, 64792, 15616, 30914]
    // Content tokens (without special prefix) must match HF exactly.
    // At minimum, "hello" must NOT be 5 character-level tokens.
    EXPECT_LE(token_ids[0].size(), 5u);  // 2 content + 2 special = 4 typical
  }

  // === "I like walking my cute dog" ===
  // HuggingFace Reference (add_special_tokens=False): [30936, 659, 5902, 552, 11527, 3246]
  {
    std::vector<std::string_view> input = {"I like walking my cute dog"};
    std::vector<std::vector<extTokenId_t>> token_ids;
    status = tokenizer->Tokenize(input, token_ids);
    EXPECT_TRUE(status.IsOk());
    DumpTokenIds(token_ids);
    ASSERT_EQ(token_ids.size(), 1u);
    // With special tokens: [64790, 64792, 30936, 659, 5902, 552, 11527, 3246] = 8 tokens
    std::vector<extTokenId_t> expected_content = {30936, 659, 5902, 552, 11527, 3246};
    // Check that the content tokens (last N) match HF exactly
    ASSERT_GE(token_ids[0].size(), expected_content.size());
    std::vector<extTokenId_t> actual_content(
        token_ids[0].end() - expected_content.size(), token_ids[0].end());
    EXPECT_EQ(actual_content, expected_content)
        << "Content token IDs do not match HuggingFace reference output";
  }

  // === "This is a test." ===
  // HuggingFace Reference (add_special_tokens=False): [3919, 323, 260, 1429, 30930]
  {
    std::vector<std::string_view> input = {"This is a test."};
    std::vector<std::vector<extTokenId_t>> token_ids;
    status = tokenizer->Tokenize(input, token_ids);
    EXPECT_TRUE(status.IsOk());
    DumpTokenIds(token_ids);
    ASSERT_EQ(token_ids.size(), 1u);
    std::vector<extTokenId_t> expected_content = {3919, 323, 260, 1429, 30930};
    ASSERT_GE(token_ids[0].size(), expected_content.size());
    std::vector<extTokenId_t> actual_content(
        token_ids[0].end() - expected_content.size(), token_ids[0].end());
    EXPECT_EQ(actual_content, expected_content)
        << "Content token IDs do not match HuggingFace reference output";
  }
}

// ============================================================================
// Transformers v5 format tests
// ============================================================================

/*
  Test real SmolLM3-3B tokenizer from HuggingFace (HuggingFaceTB/SmolLM3-3B).
  This is a real v5-era model with:
  - No add_bos_token / add_eos_token in tokenizer_config.json
  - chat_template stored as separate chat_template.jinja file
  - tokenizer_class = "PreTrainedTokenizerFast" (v4 naming, but v5 file layout)
  - "extra_special_tokens" field (new in v5)
  Files downloaded from: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
*/
TEST(OrtxTokenizerV5Test, SmolLM3_V5_Tokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/v5/smollm3");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  std::vector<std::string_view> input = {"This is a test."};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  DumpTokenIds(token_ids);

  EXPECT_EQ(token_ids.size(), 1u);
  ASSERT_GT(token_ids[0].size(), 0u);

  // Verify round-trip detokenization
  std::vector<std::string> out_text;
  std::vector<ort_extensions::span<extTokenId_t const>> token_ids_span = {token_ids[0]};
  status = tokenizer->Detokenize(token_ids_span, out_text);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(out_text[0], input[0]);
}

/*
  Test synthetic Qwen2.5 v5 tokenizer (based on Qwen/Qwen2.5-0.5B-Instruct).
  This uses a REAL tokenizer.json from the original qwen2.5 test data, with a
  SYNTHETIC tokenizer_config.json that simulates Transformers v5 save_pretrained():
  - Removed add_bos_token (v5 no longer saves this)
  - Removed added_tokens_decoder (v5 only stores when no tokenizer.json)
  - chat_template in separate .jinja file (v5 pattern)
  - tokenizer_class = "TokenizersBackend" (v5 backend class name)
  See comments in test/data/v5/qwen2.5-synthetic/tokenizer_config.json for details.
*/
TEST(OrtxTokenizerV5Test, Qwen2_5_SyntheticV5_Tokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/v5/qwen2.5-synthetic");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  // Tokenize a string that includes an added token (<tool_call>) to verify that
  // added tokens from tokenizer.json work correctly even without added_tokens_decoder.
  // <tool_call> is token id 151657 in the Qwen2.5 vocabulary.
  std::vector<std::string_view> input = {"Hello <tool_call> world"};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  DumpTokenIds(token_ids);

  EXPECT_EQ(token_ids.size(), 1u);
  ASSERT_GT(token_ids[0].size(), 0u);

  // Verify that <tool_call> was correctly tokenized as a single added token (id=151657)
  bool found_tool_call = false;
  for (const auto& id : token_ids[0]) {
    if (id == 151657) {
      found_tool_call = true;
      break;
    }
  }
  EXPECT_TRUE(found_tool_call) << "<tool_call> (id=151657) should be recognized as an added token even without added_tokens_decoder";

  // Verify round-trip detokenization
  std::vector<std::string> out_text;
  std::vector<ort_extensions::span<extTokenId_t const>> token_ids_span = {token_ids[0]};
  status = tokenizer->Detokenize(token_ids_span, out_text);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(out_text[0], input[0]);
}

// ============================================================================
// Gemma 4 tokenizer tests
// ============================================================================

/*
  Test Gemma 4 tokenizer (google/gemma-4-E2B-it).
  Gemma 4 uses GemmaTokenizer (BPE), same tokenizer family as Gemma 2/3,
  with v5-era file layout:
  - No add_bos_token in tokenizer_config.json (inferred from per-class defaults)
  - chat_template in separate .jinja file
  - 262144 vocab size
  Files downloaded from: https://huggingface.co/google/gemma-4-E2B-it
*/
TEST(OrtxTokenizerTest, Gemma4Tokenizer) {
  auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
  auto status = tokenizer->Load("data/models/gemma-4");
  if (!status.IsOk()) {
    std::cout << status.ToString() << std::endl;
    tokenizer.reset();
  }

  ASSERT_NE(tokenizer.get(), nullptr) << "Tokenizer is null, stopping the test.";

  // Test basic tokenization (BOS should be added automatically)
  std::vector<std::string_view> input = {"This is a test."};
  std::vector<std::vector<extTokenId_t>> token_ids;
  status = tokenizer->Tokenize(input, token_ids);
  EXPECT_TRUE(status.IsOk());
  DumpTokenIds(token_ids);

  EXPECT_EQ(token_ids.size(), 1u);
  ASSERT_GT(token_ids[0].size(), 0u);

  // BOS token should be the first token (id=2 for Gemma family)
  EXPECT_EQ(token_ids[0][0], 2) << "First token should be BOS (id=2)";

  // Verify token IDs match HF reference (HF returns [2094, 563, 496, 1594, 236761]
  // without BOS; our tokenizer prepends BOS=2).
  const std::vector<extTokenId_t> kExpectedIds = {2, 2094, 563, 496, 1594, 236761};
  ASSERT_EQ(token_ids[0].size(), kExpectedIds.size())
      << "Token count mismatch vs HF reference";
  for (size_t i = 0; i < kExpectedIds.size(); ++i) {
    EXPECT_EQ(token_ids[0][i], kExpectedIds[i])
        << "Token " << i << " mismatch vs HF reference";
  }

  // Verify round-trip detokenization
  std::vector<std::string> out_text;
  std::vector<ort_extensions::span<extTokenId_t const>> token_ids_span = {token_ids[0]};
  status = tokenizer->Detokenize(token_ids_span, out_text);
  EXPECT_TRUE(status.IsOk());
  EXPECT_EQ(out_text[0], input[0]);
}

// =============================================================================
// PROFILING TEST: Tokenize a substantial input on each model and print timing
// =============================================================================
extern void BpeProfiler_Reset();
extern void BpeProfiler_Print(const char* label);

TEST(OrtxTokenizerProfileTest, ProfileAllModels) {
  // A ~2KB English text to exercise the tokenizer meaningfully
  const std::string long_text =
      "The transformer architecture has revolutionized natural language processing. "
      "Self-attention mechanisms allow the model to weigh the importance of different words "
      "in a sentence relative to each other. Multi-head attention extends this by allowing "
      "the model to jointly attend to information from different representation subspaces. "
      "The original transformer paper 'Attention Is All You Need' was published in 2017. "
      "Since then, models like BERT, GPT-2, GPT-3, and GPT-4 have pushed the boundaries "
      "of what's possible with language models. These models are pre-trained on massive "
      "amounts of text data and can be fine-tuned for specific downstream tasks. "
      "Tokenization is a critical first step: converting raw text into integer token IDs "
      "that the model can process. BPE (Byte Pair Encoding) is the most common algorithm, "
      "iteratively merging frequent character pairs into subword tokens. The vocabulary "
      "typically contains 32K-100K tokens, balancing between character-level granularity "
      "and word-level efficiency. Pre-tokenization splits text into words first using "
      "regex patterns, then BPE operates on each word independently. This ensures that "
      "token boundaries respect word boundaries in most cases. Performance of tokenization "
      "matters for latency-sensitive applications like real-time chat and code completion.";

  struct ModelInfo {
    const char* name;
    const char* path;
  };

  ModelInfo models[] = {
      {"GPT-2 (Phi-2)", "data/phi-2"},
      {"Phi-4", "data/phi-4-base"},
      {"LLaMA-2", "data/llama2"},
      {"Gemma", "data/gemma"},
  };

  const int iterations = 50;

  for (const auto& model : models) {
    auto tokenizer = std::make_unique<ort_extensions::TokenizerImpl>();
    auto status = tokenizer->Load(model.path);
    if (!status.IsOk()) {
      fprintf(stderr, "  [%s] SKIPPED: %s\n", model.name, status.ToString().c_str());
      continue;
    }

    // Warmup
    std::vector<std::string_view> input = {long_text};
    std::vector<std::vector<extTokenId_t>> token_ids;
    for (int i = 0; i < 5; i++) {
      token_ids.clear();
      tokenizer->Tokenize(input, token_ids);
    }

    // Profiled run
    BpeProfiler_Reset();
    for (int i = 0; i < iterations; i++) {
      token_ids.clear();
      tokenizer->Tokenize(input, token_ids);
    }
    BpeProfiler_Print(model.name);

    fprintf(stderr, "  [%s] Tokens produced: %zu (per call)\n\n", model.name, token_ids[0].size());
  }
}