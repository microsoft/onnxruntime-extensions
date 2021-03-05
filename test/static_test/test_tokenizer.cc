// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "utils/string_utils.h"
#include "kernels/string_common.h"
#include "../tokenizer/bert_tokenizer.hpp"

TEST(tokenizer, bert_word_split) {
  ustring ind("##");
  ustring text("A AAA B BB");
  std::vector<std::u32string> words;
  KernelBertTokenizer_Split(ind, text, words);
  std::vector<std::u32string> expected{ustring("A"), ustring("AAA"), ustring("B"), ustring("BB")};
  EXPECT_EQ(expected, words);

  text = ustring("  A AAA  B BB ");
  KernelBertTokenizer_Split(ind, text, words);
  EXPECT_EQ(words, expected);
}

std::unordered_map<std::u32string, int32_t> get_vocabulary_basic() {
  std::vector<ustring> vocab_tokens = {
      ustring("[UNK]"),
      ustring("[CLS]"),
      ustring("[SEP]"),
      ustring("[PAD]"),
      ustring("[MASK]"),
      ustring("want"),
      ustring("##want"),
      ustring("##ed"),
      ustring("wa"),
      ustring("un"),
      ustring("runn"),
      ustring("##ing"),
      ustring(","),
      ustring("low"),
      ustring("lowest"),
  };
  std::unordered_map<std::u32string, int32_t> vocab;
  for (auto it = vocab_tokens.begin(); it != vocab_tokens.end(); ++it) {
    vocab[*it] = vocab.size();
  }
  return vocab;
}

TEST(tokenizer, bert_basic_tokenizer) {
  auto vocab = get_vocabulary_basic();
  std::vector<ustring> text = {ustring("UNwant\u00E9d,running")};
  std::vector<int32_t> indices;
  std::vector<int64_t> rows;
  KernelBertTokenizer_Tokenizer(vocab, ustring("##"), text, indices, rows);
  //EXPECT_EQ(indices, std::vector<int32_t>({9, 6, 7, 12, 10, 11}));
  //EXPECT_EQ(rows, std::vector<int64_t>({0, 6}));
}

std::unordered_map<std::u32string, int32_t> get_vocabulary_wordpiece() {
  std::vector<ustring> vocab_tokens = {
      ustring("[UNK]"),   // 0
      ustring("[CLS]"),   // 1
      ustring("[SEP]"),   // 2
      ustring("want"),    // 3
      ustring("##want"),  // 4
      ustring("##ed"),    // 5
      ustring("wa"),      // 6
      ustring("un"),      // 7
      ustring("runn"),    // 8
      ustring("##ing"),   // 9
  };
  std::unordered_map<std::u32string, int32_t> vocab;
  for (auto it = vocab_tokens.begin(); it != vocab_tokens.end(); ++it) {
    vocab[*it] = vocab.size();
  }
  return vocab;
}

TEST(tokenizer, bert_wordpiece_tokenizer) {
  auto vocab = get_vocabulary_wordpiece();
  std::vector<int32_t> indices;
  std::vector<int64_t> rows;

  std::vector<ustring> text = {ustring("unwanted running")};  // "un", "##want", "##ed", "runn", "##ing"
  KernelBertTokenizer_Tokenizer(vocab, ustring("##"), text, indices, rows);
  EXPECT_EQ(indices, std::vector<int32_t>({7, 4, 5, 8, 9}));
  EXPECT_EQ(rows, std::vector<int64_t>({0, 5}));

  text = std::vector<ustring>({ustring("unwantedX running")});  // "[UNK]", "runn", "##ing"
  KernelBertTokenizer_Tokenizer(vocab, ustring("##"), text, indices, rows);
  EXPECT_EQ(indices, std::vector<int32_t>({7, 4, 5, -1, 8, 9}));
  EXPECT_EQ(rows, std::vector<int64_t>({0, 6}));

  text = std::vector<ustring>({ustring("")});  //
  KernelBertTokenizer_Tokenizer(vocab, ustring("##"), text, indices, rows);
  EXPECT_EQ(indices, std::vector<int32_t>());
  EXPECT_EQ(rows, std::vector<int64_t>({0, 0}));
}
