// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <limits>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "ortx_cpp_helper.h"
#include "ortx_tokenizer.h"

namespace {

using ort_extensions::OrtxObjectPtr;

using Fixture = std::tuple<const char*, const char*, bool>;

class LiteralTokenizerFixture : public testing::Test {
 protected:
  void InitializeTokenizer(const char* path) {
    const char* keys[] = {"add_special_tokens", "skip_special_tokens"};
    const char* values[] = {"false", "false"};
    tokenizer_ = OrtxObjectPtr<OrtxTokenizer>(OrtxCreateTokenizerWithOptions, path, keys, values, 2);
    ASSERT_EQ(tokenizer_.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
  }

  void Encode(const std::string& text, bool literal, std::vector<extTokenId_t>* output) const {
    const char* input[] = {text.c_str()};
    OrtxObjectPtr<OrtxTokenId2DArray> sequences;
    const auto result = literal ? OrtxTokenizeLiteral(tokenizer_.get(), input, 1, sequences.ToBeAssigned())
                                : OrtxTokenize(tokenizer_.get(), input, 1, sequences.ToBeAssigned());
    ASSERT_EQ(result, kOrtxOK) << OrtxGetLastErrorMessage();
    const extTokenId_t* ids = nullptr;
    size_t count = 0;
    ASSERT_EQ(OrtxTokenId2DArrayGetItem(sequences.get(), 0, &ids, &count), kOrtxOK);
    output->clear();
    if (count != 0) {
      ASSERT_NE(ids, nullptr);
      output->assign(ids, ids + count);
    }
  }

  void ExpectDecodedText(const std::vector<extTokenId_t>& ids, const std::string& expected) const {
    ASSERT_FALSE(ids.empty());
    OrtxObjectPtr<OrtxStringArray> decoded;
    ASSERT_EQ(OrtxDetokenize1D(tokenizer_.get(), ids.data(), ids.size(), decoded.ToBeAssigned()), kOrtxOK)
        << OrtxGetLastErrorMessage();
    const char* text = nullptr;
    ASSERT_EQ(OrtxStringArrayGetItem(decoded.get(), 0, &text), kOrtxOK);
    ASSERT_NE(text, nullptr);
    const std::string normalized = dummy_prefix_ ? " " + expected : expected;
    EXPECT_EQ(std::string(text), normalized);
  }

  OrtxObjectPtr<OrtxTokenizer> tokenizer_;
  bool dummy_prefix_ = false;
};

class LiteralTokenizerTest : public LiteralTokenizerFixture, public testing::WithParamInterface<Fixture> {
 protected:
  void SetUp() override {
    dummy_prefix_ = std::get<2>(GetParam());
    InitializeTokenizer(std::get<0>(GetParam()));
  }
};

class ChatGLMLiteralTest : public LiteralTokenizerFixture {
 protected:
  void SetUp() override { InitializeTokenizer("data/tokenizer/THUDM/chatglm-6b"); }
};

TEST_P(LiteralTokenizerTest, PreservesOrdinaryTextAndCasing) {
  for (const char* text : {"Hello WORLD aMiXeD-case 123", "Hello-there THIS Is a Test"}) {
    SCOPED_TRACE(text);
    std::vector<extTokenId_t> legacy;
    std::vector<extTokenId_t> literal;
    ASSERT_NO_FATAL_FAILURE(Encode(text, false, &legacy));
    ASSERT_NO_FATAL_FAILURE(Encode(text, true, &literal));
    EXPECT_EQ(literal, legacy);
    ASSERT_NO_FATAL_FAILURE(ExpectDecodedText(literal, text));
  }
}

TEST_P(LiteralTokenizerTest, MarkerTextDoesNotProduceTheRegisteredId) {
  const char* marker = std::get<1>(GetParam());
  extTokenId_t marker_id = 0;
  ASSERT_EQ(OrtxConvertTokenToId(tokenizer_.get(), marker, &marker_id), kOrtxOK);
  std::vector<extTokenId_t> legacy_marker;
  ASSERT_NO_FATAL_FAILURE(Encode(marker, false, &legacy_marker));
  ASSERT_NE(std::find(legacy_marker.begin(), legacy_marker.end(), marker_id), legacy_marker.end());

  const std::string text = std::string("Hello ") + marker + " WORLD";
  std::vector<extTokenId_t> literal;
  ASSERT_NO_FATAL_FAILURE(Encode(text, true, &literal));
  EXPECT_EQ(std::find(literal.begin(), literal.end(), marker_id), literal.end());
  ASSERT_NO_FATAL_FAILURE(ExpectDecodedText(literal, text));
}

TEST_P(LiteralTokenizerTest, EmptyLiteralHasNoAutomaticTokens) {
  const char* keys[] = {"add_special_tokens"};
  const char* values[] = {"true"};
  ASSERT_EQ(OrtxUpdateTokenizerOptions(tokenizer_.get(), keys, values, 1), kOrtxOK);
  std::vector<extTokenId_t> ids;
  ASSERT_NO_FATAL_FAILURE(Encode("", true, &ids));
  EXPECT_TRUE(ids.empty());
}

TEST_P(LiteralTokenizerTest, RepeatedCallsPreserveLegacyResults) {
  const std::string first = "Hello-there THIS Is a Test";
  const std::string second = "aMiXeD Case After UPPER WORDS";
  std::vector<extTokenId_t> expected_first;
  std::vector<extTokenId_t> expected_second;
  ASSERT_NO_FATAL_FAILURE(Encode(first, false, &expected_first));
  ASSERT_NO_FATAL_FAILURE(Encode(second, false, &expected_second));
  for (int i = 0; i < 8; ++i) {
    std::vector<extTokenId_t> ids;
    ASSERT_NO_FATAL_FAILURE(Encode(first, true, &ids));
    EXPECT_EQ(ids, expected_first);
    ASSERT_NO_FATAL_FAILURE(Encode(second, false, &ids));
    EXPECT_EQ(ids, expected_second);
  }
}

TEST_P(LiteralTokenizerTest, ConcurrentLiteralAndLegacyCallsAreIndependent) {
  const std::string text = "Hello-there THIS Is a Test";
  std::vector<extTokenId_t> expected;
  ASSERT_NO_FATAL_FAILURE(Encode(text, false, &expected));
  auto worker = [&](bool literal) {
    for (int i = 0; i < 16; ++i) {
      std::vector<extTokenId_t> ids;
      ASSERT_NO_FATAL_FAILURE(Encode(text, literal, &ids));
      EXPECT_EQ(ids, expected);
    }
  };
  std::thread literal(worker, true);
  std::thread legacy(worker, false);
  literal.join();
  legacy.join();
}

TEST_P(LiteralTokenizerTest, InvalidUtf8LeavesOutputUnassigned) {
  for (const char* text : {"\xff", "control\xff", "\xc0\xaf"}) {
    const char* input[] = {text};
    OrtxTokenId2DArray* output = nullptr;
    EXPECT_EQ(OrtxTokenizeLiteral(tokenizer_.get(), input, 1, &output), kOrtxErrorInvalidArgument);
    EXPECT_EQ(output, nullptr);
    EXPECT_NE(std::string(OrtxGetLastErrorMessage()).size(), 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(TokenizerFamilies, LiteralTokenizerTest,
                         testing::Values(Fixture{"data/phi-3", "<|assistant|>", true},
                                         Fixture{"data/tokenizer/fairseq/xlm-roberta-base", "</s>", false},
                                         Fixture{"data/tokenizer/nmt", "</s>", false},
                                         Fixture{"data/tokenizer/nmt", "(#SPLIT)", false}));

TEST(LiteralTokenizerArgumentsTest, RejectsNullArguments) {
  OrtxTokenId2DArray* output = nullptr;
  const char* input[] = {"text"};
  EXPECT_EQ(OrtxTokenizeLiteral(nullptr, input, 1, &output), kOrtxErrorInvalidArgument);
  EXPECT_EQ(output, nullptr);
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/tokenizer/nmt");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK);
  EXPECT_EQ(OrtxTokenizeLiteral(tokenizer.get(), nullptr, 1, &output), kOrtxErrorInvalidArgument);
  EXPECT_EQ(OrtxTokenizeLiteral(tokenizer.get(), input, 1, nullptr), kOrtxErrorInvalidArgument);
  const char* null_input[] = {nullptr};
  EXPECT_EQ(OrtxTokenizeLiteral(tokenizer.get(), null_input, 1, &output), kOrtxErrorInvalidArgument);
  EXPECT_EQ(output, nullptr);
}

TEST(LiteralTokenizerArgumentsTest, RejectsUnrepresentableSparseVocabulary) {
  OrtxObjectPtr<OrtxTokenizer> tokenizer(OrtxCreateTokenizer, "data/added-tokens");
  ASSERT_EQ(tokenizer.Code(), kOrtxOK) << OrtxGetLastErrorMessage();
  const char* input[] = {"Hello WORLD aMiXeD-case 123"};
  OrtxTokenId2DArray* output = nullptr;
  EXPECT_EQ(OrtxTokenizeLiteral(tokenizer.get(), input, 1, &output), kOrtxErrorInvalidArgument);
  EXPECT_EQ(output, nullptr);
  EXPECT_NE(std::string(OrtxGetLastErrorMessage()).size(), 0u);
}

TEST_F(LiteralTokenizerFixture, RejectsUnknownFallbackWithoutConfiguredId) {
  ASSERT_NO_FATAL_FAILURE(InitializeTokenizer("data/unigram-no-unk"));

  const auto sentinel = std::numeric_limits<extTokenId_t>::max();
  std::vector<extTokenId_t> expected_unknown;
  ASSERT_NO_FATAL_FAILURE(Encode("b", false, &expected_unknown));
  ASSERT_NE(std::find(expected_unknown.begin(), expected_unknown.end(), sentinel), expected_unknown.end());
  const char* input[] = {"b"};
  OrtxObjectPtr<OrtxTokenId2DArray> output;
  EXPECT_EQ(OrtxTokenizeLiteral(tokenizer_.get(), input, 1, output.ToBeAssigned()), kOrtxErrorInvalidArgument);
  EXPECT_EQ(output.get(), nullptr);
  EXPECT_NE(std::string(OrtxGetLastErrorMessage()).size(), 0u);
  std::vector<extTokenId_t> legacy;
  ASSERT_NO_FATAL_FAILURE(Encode("b", false, &legacy));
  EXPECT_EQ(legacy, expected_unknown);

  std::vector<extTokenId_t> expected_text;
  std::vector<extTokenId_t> literal;
  ASSERT_NO_FATAL_FAILURE(Encode("a", false, &expected_text));
  ASSERT_NO_FATAL_FAILURE(Encode("a", true, &literal));
  EXPECT_EQ(literal, expected_text);
  EXPECT_EQ(std::find(literal.begin(), literal.end(), sentinel), literal.end());
  ASSERT_NO_FATAL_FAILURE(Encode("", true, &literal));
  EXPECT_TRUE(literal.empty());
}

TEST_F(ChatGLMLiteralTest, LiteralDoesNotAppendAutomaticEndings) {
  std::vector<extTokenId_t> legacy;
  std::vector<extTokenId_t> literal;
  ASSERT_NO_FATAL_FAILURE(Encode("Hello world", false, &legacy));
  ASSERT_NO_FATAL_FAILURE(Encode("Hello world", true, &literal));
  extTokenId_t gmask = 0;
  extTokenId_t sop = 0;
  ASSERT_EQ(OrtxConvertTokenToId(tokenizer_.get(), "[gMASK]", &gmask), kOrtxOK);
  ASSERT_EQ(OrtxConvertTokenToId(tokenizer_.get(), "<sop>", &sop), kOrtxOK);
  ASSERT_GE(legacy.size(), 2u);
  EXPECT_EQ(legacy[legacy.size() - 2], gmask);
  EXPECT_EQ(legacy.back(), sop);
  EXPECT_EQ(std::find(literal.begin(), literal.end(), gmask), literal.end());
  EXPECT_EQ(std::find(literal.begin(), literal.end(), sop), literal.end());
}

TEST_F(ChatGLMLiteralTest, VocabularyControlMarkersRemainLiteralWithoutAddedTokenMetadata) {
  for (const char* marker : {"<unk>", "[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>"}) {
    SCOPED_TRACE(marker);
    extTokenId_t control_id = 0;
    ASSERT_EQ(OrtxConvertTokenToId(tokenizer_.get(), marker, &control_id), kOrtxOK);
    std::vector<extTokenId_t> literal;
    ASSERT_NO_FATAL_FAILURE(Encode(std::string("Hello ") + marker + " world", true, &literal));
    EXPECT_EQ(std::find(literal.begin(), literal.end(), control_id), literal.end());
  }
}

TEST_F(ChatGLMLiteralTest, RejectsUnrepresentableInputWithoutChangingLegacyResults) {
  for (const char* text : {"Hello \xf4\x8f\xbf\xbf world", "Hello \xf0\x9f\xab\xa8 world"}) {
    SCOPED_TRACE(text);
    std::vector<extTokenId_t> expected;
    ASSERT_NO_FATAL_FAILURE(Encode(text, false, &expected));
    ASSERT_FALSE(expected.empty());
    const char* input[] = {text};
    OrtxObjectPtr<OrtxTokenId2DArray> output;
    EXPECT_EQ(OrtxTokenizeLiteral(tokenizer_.get(), input, 1, output.ToBeAssigned()), kOrtxErrorInvalidArgument);
    EXPECT_EQ(output.get(), nullptr);
    EXPECT_NE(std::string(OrtxGetLastErrorMessage()).size(), 0u);
    std::vector<extTokenId_t> legacy;
    ASSERT_NO_FATAL_FAILURE(Encode(text, false, &legacy));
    EXPECT_EQ(legacy, expected);
  }
}

TEST_F(ChatGLMLiteralTest, LiteralRequestsPreserveLegacyEndingBehavior) {
  std::vector<extTokenId_t> expected;
  ASSERT_NO_FATAL_FAILURE(Encode("Hello world", false, &expected));
  for (int i = 0; i < 8; ++i) {
    std::vector<extTokenId_t> literal;
    std::vector<extTokenId_t> legacy;
    ASSERT_NO_FATAL_FAILURE(Encode("Hello world", true, &literal));
    ASSERT_NO_FATAL_FAILURE(Encode("Hello world", false, &legacy));
    EXPECT_EQ(legacy, expected);
  }
}

}  // namespace
