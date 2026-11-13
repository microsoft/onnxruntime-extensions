// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <chrono>
#include <cstdlib>
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

TEST(TikTokenPathValidationTest, AcceptsEnclosedRelativePathsAndExtensionlessVocab) {
  const auto base_config = ReadTokenizerFile("data/tokenizer/phi-3-small/tokenizer_config.json");
  const auto base_module = json::parse(ReadTokenizerFile("data/tokenizer/phi-3-small/tokenizer_module.json"));
  const auto raw_tiktoken = ReadTokenizerFile("data/tokenizer/phi-3-small/cl100k_base.tiktoken");
  const std::vector<extTokenId_t> expected_ids = {2028, 374, 264, 1296, 13};

  ScopedTokenizerDirectory sandbox;
  const auto tok_dir = sandbox.path / "model.v1";
  std::filesystem::create_directories(tok_dir / "sub");

  auto write_file = [](const std::filesystem::path& file_path, const std::string& content) {
    std::ofstream out(file_path, std::ios::binary);
    out << content;
  };

  write_file(tok_dir / "tokenizer_config.json", base_config);
  write_file(tok_dir / "cl100k_base.tiktoken", raw_tiktoken);
  write_file(tok_dir / "sub" / "nested.tiktoken", raw_tiktoken);
  write_file(tok_dir / "tiktoken_vocab", raw_tiktoken);

  for (const char* rel_path :
       {"cl100k_base.tiktoken", "./cl100k_base.tiktoken", "sub/nested.tiktoken", "tiktoken_vocab"}) {
    SCOPED_TRACE(rel_path);
    json mod = base_module;
    mod["tiktoken_file"] = rel_path;
    write_file(tok_dir / "tokenizer_module.json", mod.dump());

    OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, tok_dir.string().c_str());
    ASSERT_EQ(tokenizer.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
    std::vector<extTokenId_t> actual_ids;
    ASSERT_NO_FATAL_FAILURE(Encode(tokenizer.get(), "This is a test.", &actual_ids));
    EXPECT_EQ(actual_ids, expected_ids);
  }

#ifdef _WIN32
  {
    std::string lower_dir = tok_dir.string();
    for (char& ch : lower_dir) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, lower_dir.c_str());
    ASSERT_EQ(tokenizer.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
  }
#endif

  // Verified vocabulary bytes are retained from the opened handle rather than reopened by pathname.
  {
    json mod = base_module;
    mod["tiktoken_file"] = "cl100k_base.tiktoken";
    write_file(tok_dir / "tokenizer_module.json", mod.dump());
    ort_extensions::TokenJsonConfig cfg;
    ASSERT_TRUE(cfg.Load(tok_dir.string()).IsOk());
    std::filesystem::remove(tok_dir / "cl100k_base.tiktoken");

    std::unique_ptr<std::istream> vocab_stream;
    ASSERT_TRUE(cfg.OpenVocabFile(vocab_stream).IsOk());
    ASSERT_NE(vocab_stream, nullptr);
    const std::string loaded_bytes{std::istreambuf_iterator<char>{*vocab_stream}, {}};
    EXPECT_EQ(loaded_bytes, raw_tiktoken);
  }

  // Rooted file paths preserve their root directory instead of collapsing parent_path() to empty.
  EXPECT_EQ(ort_extensions::path("tokenizer.json").parent_path(), "");
  EXPECT_EQ(ort_extensions::path("/tokenizer.json").parent_path(), "/");
  EXPECT_EQ((ort_extensions::path("/") / "tokenizer_config.json").string(), "/tokenizer_config.json");
#ifdef _WIN32
  EXPECT_EQ(ort_extensions::path("C:\\tokenizer.json").parent_path(), "C:\\");
  EXPECT_EQ((ort_extensions::path("C:\\") / "tokenizer_config.json").string(), "C:\\tokenizer_config.json");
#endif
}

TEST(TikTokenPathValidationTest, RejectsTraversalAbsoluteDeviceDirectoryAndEscapingLinks) {
  const auto base_config = json::parse(ReadTokenizerFile("data/tokenizer/phi-3-small/tokenizer_config.json"));
  const auto base_module = json::parse(ReadTokenizerFile("data/tokenizer/phi-3-small/tokenizer_module.json"));
  const auto raw_tiktoken = ReadTokenizerFile("data/tokenizer/phi-3-small/cl100k_base.tiktoken");

  ScopedTokenizerDirectory sandbox;
  const auto tok_dir = sandbox.path / "tok";
  const auto outside_dir = sandbox.path / "tok_outside";
  std::filesystem::create_directories(tok_dir / "sub");
  std::filesystem::create_directories(outside_dir);

  auto write_file = [](const std::filesystem::path& file_path, const std::string& content) {
    std::ofstream out(file_path, std::ios::binary);
    out << content;
  };

  const auto outside_file = outside_dir / "outside.tiktoken";
  write_file(outside_file, raw_tiktoken);
  write_file(sandbox.path / "parent.tiktoken", raw_tiktoken);
  write_file(tok_dir / "cl100k_base.tiktoken", raw_tiktoken);

  std::vector<std::string> rejected_paths = {
      "..",
      "../parent.tiktoken",
      "..\\parent.tiktoken",
      "sub/../../parent.tiktoken",
      "../tok_outside/outside.tiktoken",
      outside_file.string(),
      "/etc/hostname",
      "\\Windows\\win.ini",
      "C:\\Windows\\win.ini",
      "C:parent.tiktoken",
      ".",
      "./.",
      "sub",
      "sub/",
      "sub/.",
      "sub//cl100k_base.tiktoken",
  };
  rejected_paths.emplace_back("cl100k_base.tiktoken\0ignored", 28);

#ifdef _WIN32
  rejected_paths.push_back("cl100k_base.tiktoken:$DATA");
  rejected_paths.push_back("NUL");
  rejected_paths.push_back("CON.tiktoken");
  rejected_paths.push_back(".. ");
  rejected_paths.push_back("cl100k_base.tiktoken.");

  const auto junction_dir = tok_dir / "linked_out";
  const std::wstring mklink_cmd =
      L"cmd.exe /c mklink /J \"" + junction_dir.wstring() + L"\" \"" + outside_dir.wstring() + L"\" >nul 2>&1";
  ASSERT_EQ(_wsystem(mklink_cmd.c_str()), 0);
  ASSERT_TRUE(std::filesystem::exists(junction_dir / "outside.tiktoken"));
  rejected_paths.push_back("linked_out/outside.tiktoken");
#else
  std::error_code symlink_ec;
  std::filesystem::create_symlink(outside_file, tok_dir / "escaped.tiktoken", symlink_ec);
  ASSERT_FALSE(symlink_ec) << symlink_ec.message();
  rejected_paths.push_back("escaped.tiktoken");
  std::filesystem::create_directory_symlink(outside_dir, tok_dir / "linked_out", symlink_ec);
  ASSERT_FALSE(symlink_ec) << symlink_ec.message();
  rejected_paths.push_back("linked_out/outside.tiktoken");
#endif

  // 1. Via tokenizer_config.json (without tokenizer_module.json override)
  for (const auto& bad_path : rejected_paths) {
    SCOPED_TRACE(bad_path);
    json cfg = base_config;
    cfg["tiktoken_file"] = bad_path;
    const auto cfg_bytes = cfg.dump();
    write_file(tok_dir / "tokenizer_config.json", cfg_bytes);
    std::error_code ec;
    std::filesystem::remove(tok_dir / "tokenizer_module.json", ec);

    OrtxTokenizer* tokenizer = nullptr;
    EXPECT_EQ(OrtxCreateTokenizer(&tokenizer, tok_dir.string().c_str()), kOrtxErrorInvalidFile);
    EXPECT_EQ(tokenizer, nullptr);

    if (ort_extensions::path::is_safe_relative_path(bad_path)) {
      // Link escapes are filesystem-level; lexical check passes for blob mode which does not touch disk.
      continue;
    }
    const OrtxTokenizerBlob blob(cfg_bytes, "", "", raw_tiktoken);
    EXPECT_EQ(OrtxCreateTokenizerFromBlob(&tokenizer, &blob), kOrtxErrorInvalidFile);
    EXPECT_EQ(tokenizer, nullptr);
  }

  // 2. Via tokenizer_module.json override
  write_file(tok_dir / "tokenizer_config.json", base_config.dump());
  json bad_mod = base_module;
  bad_mod["tiktoken_file"] = "../parent.tiktoken";
  write_file(tok_dir / "tokenizer_module.json", bad_mod.dump());
  {
    OrtxTokenizer* tokenizer = nullptr;
    EXPECT_EQ(OrtxCreateTokenizer(&tokenizer, tok_dir.string().c_str()), kOrtxErrorInvalidFile);
    EXPECT_EQ(tokenizer, nullptr);
  }

#ifdef _WIN32
  std::error_code rm_ec;
  std::filesystem::remove(tok_dir / "linked_out", rm_ec);
#endif
}

TEST(TikTokenPathValidationTest, HandlesNullAndRejectsNonStringTikTokenField) {
  const auto config_str = ReadTokenizerFile("data/added-tokens/tokenizer_config.json");
  const auto vocab_str = ReadTokenizerFile("data/added-tokens/tokenizer.json");

  ScopedTokenizerDirectory dir;
  dir.Write("tokenizer.json", vocab_str);

  // Explicit null tiktoken_file falls back to tokenizer.json cleanly.
  auto null_cfg = json::parse(config_str);
  null_cfg["tiktoken_file"] = nullptr;
  const auto null_cfg_bytes = null_cfg.dump();
  dir.Write("tokenizer_config.json", null_cfg_bytes);
  {
    OrtxObjectPtr<OrtxTokenizer> file_tok(OrtxCreateTokenizer, dir.path.string().c_str());
    EXPECT_EQ(file_tok.Code(), kOrtxOK) << OrtxGetLastErrorMessage();

    const OrtxTokenizerBlob blob(null_cfg_bytes, vocab_str);
    OrtxObjectPtr<OrtxTokenizer> blob_tok(OrtxCreateTokenizerFromBlob, &blob);
    EXPECT_EQ(blob_tok.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
  }

  // Non-string tiktoken_file fails cleanly with kOrtxErrorInvalidArgument without throwing.
  for (const auto* bad_json : {R"({"tokenizer_class":"GPT2Tokenizer","tiktoken_file":123})",
                               R"({"tokenizer_class":"GPT2Tokenizer","tiktoken_file":["cl100k_base.tiktoken"]})"}) {
    SCOPED_TRACE(bad_json);
    const std::string bad_json_bytes = bad_json;
    dir.Write("tokenizer_config.json", bad_json_bytes);
    OrtxTokenizer* tokenizer = nullptr;
    EXPECT_EQ(OrtxCreateTokenizer(&tokenizer, dir.path.string().c_str()), kOrtxErrorInvalidArgument);
    EXPECT_EQ(tokenizer, nullptr);

    const OrtxTokenizerBlob blob(bad_json_bytes, vocab_str);
    EXPECT_EQ(OrtxCreateTokenizerFromBlob(&tokenizer, &blob), kOrtxErrorInvalidArgument);
    EXPECT_EQ(tokenizer, nullptr);
  }
}

}  // namespace
