// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <chrono>
#include <filesystem>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "ortx_cpp_helper.h"
#include "ortx_tokenizer.h"
#include "tokenizer_jsconfig.hpp"

namespace {

using ort_extensions::OrtxObjectPtr;
using json = nlohmann::json;

std::string ReadTokenizerFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open tokenizer fixture: " + path);
  }
  std::string result{std::istreambuf_iterator<char>{input}, {}};
  if (input.bad() || result.empty()) {
    throw std::runtime_error("Cannot read tokenizer fixture: " + path);
  }
  return result;
}

void Encode(OrtxTokenizer* tokenizer, const char* text, std::vector<extTokenId_t>* output) {
  const char* input[] = {text};
  OrtxObjectPtr<OrtxTokenId2DArray> tokens;
  ASSERT_EQ(OrtxTokenize(tokenizer, input, 1, tokens.ToBeAssigned()), kOrtxOK) << OrtxGetLastErrorMessage();
  const extTokenId_t* ids = nullptr;
  size_t count = 0;
  ASSERT_EQ(OrtxTokenId2DArrayGetItem(tokens.get(), 0, &ids, &count), kOrtxOK);
  ASSERT_GT(count, 0u);
  ASSERT_NE(ids, nullptr);
  output->assign(ids, ids + count);
}

class BlobTokenizerParityTest : public testing::TestWithParam<const char*> {};

struct ScopedTokenizerDirectory {
  ScopedTokenizerDirectory()
      : path{
            std::filesystem::temp_directory_path() /
            ("ortx-invalid-tokenizer-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))} {
    if (!std::filesystem::create_directory(path)) {
      throw std::runtime_error("Cannot create tokenizer test directory");
    }
  }
  ~ScopedTokenizerDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
    EXPECT_FALSE(error) << error.message();
  }
  void Write(const char* name, const std::string& value) const {
    std::ofstream output;
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(path / name, std::ios::binary);
    output << value;
  }
  const std::filesystem::path path;
};

TEST_P(BlobTokenizerParityTest, MatchesPathTokenization) {
  const std::string path = GetParam();
  const auto config = ReadTokenizerFile(path + "/tokenizer_config.json");
  const auto vocabulary = ReadTokenizerFile(path + "/tokenizer.json");
  const OrtxTokenizerBlob blob(config, vocabulary);
  OrtxObjectPtr<OrtxTokenizer> memory(OrtxCreateTokenizerFromBlob, &blob);
  ASSERT_EQ(memory.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
  OrtxObjectPtr<OrtxTokenizer> file(OrtxCreateTokenizer, path.c_str());
  ASSERT_EQ(file.Code(), kOrtxOK) << OrtxGetLastErrorMessage();

  for (const char* text : {"This is a test.", "Hello WORLD 123"}) {
    SCOPED_TRACE(text);
    std::vector<extTokenId_t> expected;
    std::vector<extTokenId_t> actual;
    ASSERT_NO_FATAL_FAILURE(Encode(file.get(), text, &expected));
    ASSERT_NO_FATAL_FAILURE(Encode(memory.get(), text, &actual));
    EXPECT_EQ(actual, expected);
  }
}

INSTANTIATE_TEST_SUITE_P(TokenizerFamilies, BlobTokenizerParityTest,
                         testing::Values("data/added-tokens", "data/tokenizer/nmt",
                                         "data/tokenizer/fairseq/xlm-roberta-base"));

TEST(BlobTokenizerTest, BpeVocabularyWithUnigramClassLoadsWithoutAPath) {
  auto config = json::parse(ReadTokenizerFile("data/added-tokens/tokenizer_config.json"));
  config["tokenizer_class"] = "ChatGLMTokenizer";
  const auto config_bytes = config.dump();
  const auto vocabulary = ReadTokenizerFile("data/added-tokens/tokenizer.json");
  const OrtxTokenizerBlob blob(config_bytes, vocabulary);
  OrtxObjectPtr<OrtxTokenizer> memory(OrtxCreateTokenizerFromBlob, &blob);
  ASSERT_EQ(memory.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
  OrtxObjectPtr<OrtxTokenizer> file(OrtxCreateTokenizer, "data/added-tokens");
  ASSERT_EQ(file.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
  const char* keys[] = {"add_special_tokens"};
  const char* values[] = {"false"};
  for (auto* tokenizer : {file.get(), memory.get()}) {
    ASSERT_EQ(OrtxUpdateTokenizerOptions(tokenizer, keys, values, 1), kOrtxOK);
  }
  std::vector<extTokenId_t> expected;
  std::vector<extTokenId_t> actual;
  ASSERT_NO_FATAL_FAILURE(Encode(file.get(), "<|user|>", &expected));
  ASSERT_NO_FATAL_FAILURE(Encode(memory.get(), "<|user|>", &actual));
  ASSERT_EQ(expected, std::vector<extTokenId_t>{32010});
  EXPECT_EQ(actual, expected);
}

TEST(BlobTokenizerTest, ConfigOnlyAddedTokenMatchesPathLookupAndEncoding) {
  constexpr const char* token = "<|assistant|>";
  constexpr extTokenId_t token_id = 32001;
  const auto config = ReadTokenizerFile("data/added-tokens/tokenizer_config.json");
  const auto vocabulary = ReadTokenizerFile("data/added-tokens/tokenizer.json");
  const auto parsed_vocabulary = json::parse(vocabulary);
  for (const auto& added_token : parsed_vocabulary.at("added_tokens")) {
    ASSERT_NE(added_token.at("content").get<std::string>(), token);
  }
  const OrtxTokenizerBlob blob(config, vocabulary);
  OrtxObjectPtr<OrtxTokenizer> memory(OrtxCreateTokenizerFromBlob, &blob);
  ASSERT_EQ(memory.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
  OrtxObjectPtr<OrtxTokenizer> file(OrtxCreateTokenizer, "data/added-tokens");
  ASSERT_EQ(file.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
  for (auto* tokenizer : {file.get(), memory.get()}) {
    extTokenId_t actual_id = 0;
    ASSERT_EQ(OrtxConvertTokenToId(tokenizer, token, &actual_id), kOrtxOK) << OrtxGetLastErrorMessage();
    EXPECT_EQ(actual_id, token_id);
    const char* keys[] = {"add_special_tokens"};
    const char* values[] = {"false"};
    ASSERT_EQ(OrtxUpdateTokenizerOptions(tokenizer, keys, values, 1), kOrtxOK);
    std::vector<extTokenId_t> actual;
    ASSERT_NO_FATAL_FAILURE(Encode(tokenizer, token, &actual));
    EXPECT_EQ(actual, std::vector<extTokenId_t>{token_id});
  }
}

TEST(BlobTokenizerTest, MalformedJsonReturnsStatusWithoutEscapingTheCApi) {
  const auto config = ReadTokenizerFile("data/added-tokens/tokenizer_config.json");
  const auto vocabulary = ReadTokenizerFile("data/added-tokens/tokenizer.json");
  const std::string malformed = "{";
  const std::string wrong_type = R"({"tokenizer_class":17})";
  const std::vector<std::pair<std::string, std::string>> inputs{
      {malformed, vocabulary}, {config, malformed}, {wrong_type, vocabulary}};
  for (size_t i = 0; i < inputs.size(); ++i) {
    SCOPED_TRACE(i);
    const OrtxTokenizerBlob blob(inputs[i].first, inputs[i].second);
    OrtxTokenizer* tokenizer = nullptr;
    extError_t code = kOrtxOK;
    ASSERT_NO_THROW(code = OrtxCreateTokenizerFromBlob(&tokenizer, &blob));
    EXPECT_EQ(code, kOrtxErrorInvalidArgument);
    EXPECT_EQ(tokenizer, nullptr);
    EXPECT_NE(std::string{OrtxGetLastErrorMessage()}, "");
  }
}

TEST(BlobTokenizerTest, MalformedFileJsonReturnsStatusWithoutEscapingTheCApi) {
  const auto config = ReadTokenizerFile("data/added-tokens/tokenizer_config.json");
  const auto vocabulary = ReadTokenizerFile("data/added-tokens/tokenizer.json");
  for (bool invalid_config : {false, true}) {
    SCOPED_TRACE(invalid_config);
    ScopedTokenizerDirectory directory;
    directory.Write("tokenizer_config.json", invalid_config ? "{" : config);
    directory.Write("tokenizer.json", invalid_config ? vocabulary : "{");
    const auto path = directory.path.string();
    OrtxTokenizer* tokenizer = nullptr;
    extError_t code = kOrtxOK;
    ASSERT_NO_THROW(code = OrtxCreateTokenizer(&tokenizer, path.c_str()));
    EXPECT_EQ(code, kOrtxErrorInvalidArgument);
    EXPECT_EQ(tokenizer, nullptr);
    EXPECT_NE(std::string{OrtxGetLastErrorMessage()}, "");
  }
}

TEST(BlobTokenizerTest, FailedCreationClearsOutputAndRejectsNullOutput) {
  const auto config = ReadTokenizerFile("data/added-tokens/tokenizer_config.json");
  const auto vocabulary = ReadTokenizerFile("data/added-tokens/tokenizer.json");
  const OrtxTokenizerBlob blob(config, vocabulary);
  OrtxObjectPtr<OrtxTokenizer> owned(OrtxCreateTokenizerFromBlob, &blob);
  ASSERT_EQ(owned.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
  auto* output = owned.get();
  EXPECT_EQ(OrtxCreateTokenizerFromBlob(&output, nullptr), kOrtxErrorInvalidArgument);
  EXPECT_EQ(output, nullptr);
  output = owned.get();
  EXPECT_EQ(OrtxCreateTokenizer(&output, nullptr), kOrtxErrorInvalidArgument);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(OrtxCreateTokenizerFromBlob(nullptr, &blob), kOrtxErrorInvalidArgument);
  EXPECT_EQ(OrtxCreateTokenizer(nullptr, "data/added-tokens"), kOrtxErrorInvalidArgument);
}

TEST(BlobTokenizerConfigTest, MissingOrNonObjectAddedTokensUseEmptyMetadata) {
  const std::string vocabulary = "{}";
  for (const auto* config_bytes :
       {R"({"tokenizer_class":"GPT2Tokenizer"})", R"({"tokenizer_class":"GPT2Tokenizer","added_tokens_decoder":null})",
        R"({"tokenizer_class":"GPT2Tokenizer","added_tokens_decoder":[]})"}) {
    SCOPED_TRACE(config_bytes);
    const OrtxTokenizerBlob blob(config_bytes, vocabulary);
    ort_extensions::TokenJsonConfig config;
    const auto status = config.LoadFromBlob(blob);
    ASSERT_TRUE(status.IsOk()) << status.ToString();
    ASSERT_NE(config.added_tokens_decoder, nullptr);
    EXPECT_TRUE(config.added_tokens_decoder->is_object());
    EXPECT_TRUE(config.added_tokens_decoder->empty());
  }
}

TEST(BlobTokenizerConfigTest, CapturesAddedTokensBeforeModuleOverrides) {
  const std::string config_bytes =
      R"({"tokenizer_class":"GPT2Tokenizer","added_tokens_decoder":{"7":{"content":"<config>"}}})";
  const std::string vocabulary = "{}";
  const std::string module =
      R"({"added_tokens_decoder":{"8":{"content":"<override>"}},"added_tokens":[{"id":8,"content":"<module>"}]})";
  const OrtxTokenizerBlob blob(config_bytes, vocabulary, module);
  ort_extensions::TokenJsonConfig config;
  const auto status = config.LoadFromBlob(blob);
  ASSERT_TRUE(status.IsOk()) << status.ToString();
  ASSERT_NE(config.added_tokens_decoder, nullptr);
  EXPECT_EQ(*config.added_tokens_decoder, json::parse(config_bytes).at("added_tokens_decoder"));
  EXPECT_EQ(config.added_tokens_.at(ustring("<module>")).id_, 8);
}

}  // namespace
