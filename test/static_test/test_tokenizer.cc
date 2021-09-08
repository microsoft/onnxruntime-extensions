// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "string_utils.h"
#include "wordpiece_tokenizer.hpp"
#include "bert_tokenizer.hpp"

TEST(tokenizer, bert_word_split) {
  ustring ind("##");
  ustring text("A AAA B BB");
  std::vector<std::u32string> words;
  KernelWordpieceTokenizer_Split(ind, text, words);
  std::vector<std::u32string> expected{ustring("A"), ustring("AAA"), ustring("B"), ustring("BB")};
  EXPECT_EQ(expected, words);

  text = ustring("  A AAA  B BB ");
  KernelWordpieceTokenizer_Split(ind, text, words);
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

std::vector<ustring> ustring_vector_convertor(std::vector<std::string> input) {
  std::vector<ustring> result;
  for (const auto& str : input) {
    result.emplace_back(str);
  }
  return result;
}

TEST(tokenizer, wordpiece_basic_tokenizer) {
  auto vocab = get_vocabulary_basic();
  std::vector<ustring> text = {ustring("UNwant\u00E9d,running")};
  std::vector<ustring> tokens;
  std::vector<int32_t> indices;
  std::vector<int64_t> rows;
  KernelWordpieceTokenizer_Tokenizer(vocab, ustring("##"), ustring("[unk]"), text, tokens, indices, rows);
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

TEST(tokenizer, wordpiece_wordpiece_tokenizer) {
  auto vocab = get_vocabulary_wordpiece();
  std::vector<int32_t> indices;
  std::vector<int64_t> rows;
  std::vector<ustring> tokens;

  std::vector<ustring> text = {ustring("unwanted running")};  // "un", "##want", "##ed", "runn", "##ing"
  KernelWordpieceTokenizer_Tokenizer(vocab, ustring("##"), ustring("[UNK]"), text, tokens, indices, rows);
  EXPECT_EQ(tokens, std::vector<ustring>({ustring("un"), ustring("##want"), ustring("##ed"),
                                          ustring("runn"), ustring("##ing")}));
  EXPECT_EQ(indices, std::vector<int32_t>({7, 4, 5, 8, 9}));
  EXPECT_EQ(rows, std::vector<int64_t>({0, 5}));

  text = std::vector<ustring>({ustring("unwantedX running")});  // "[UNK]", "runn", "##ing"
  KernelWordpieceTokenizer_Tokenizer(vocab, ustring("##"), ustring("[UNK]"), text, tokens, indices, rows);
  EXPECT_EQ(tokens, std::vector<ustring>({ustring("un"), ustring("##want"), ustring("##ed"),
                                          ustring("[UNK]"), ustring("runn"), ustring("##ing")}));
  EXPECT_EQ(indices, std::vector<int32_t>({7, 4, 5, -1, 8, 9}));
  EXPECT_EQ(rows, std::vector<int64_t>({0, 6}));

  text = std::vector<ustring>({ustring("")});  //
  KernelWordpieceTokenizer_Tokenizer(vocab, ustring("##"), ustring("[unk]"), text, tokens, indices, rows);
  EXPECT_EQ(tokens, std::vector<ustring>());
  EXPECT_EQ(indices, std::vector<int32_t>());
  EXPECT_EQ(rows, std::vector<int64_t>({0, 0}));
}

TEST(tokenizer, bert_wordpiece_tokenizer_rows) {
  auto vocab = get_vocabulary_wordpiece();
  std::vector<int32_t> indices;
  std::vector<int64_t> rows;
  std::vector<ustring> tokens;

  std::vector<int64_t> existing_indices({0, 2, 3});
  std::vector<ustring> text = {ustring("unwanted"), ustring("running"), ustring("running")};
  KernelWordpieceTokenizer_Tokenizer(vocab, ustring("##"), ustring("[UNK]"), text, tokens, indices, rows,
                                     existing_indices.data(), existing_indices.size());
  EXPECT_EQ(tokens, std::vector<ustring>({ustring("un"), ustring("##want"), ustring("##ed"),
                                          ustring("runn"), ustring("##ing"),
                                          ustring("runn"), ustring("##ing")}));
  EXPECT_EQ(indices, std::vector<int32_t>({7, 4, 5, 8, 9, 8, 9}));
  EXPECT_EQ(rows, std::vector<int64_t>({0, 5, 7}));
}

TEST(tokenizer, basic_tokenizer_chinese) {
  ustring test_case = ustring("ÀÁÂÃÄÅÇÈÉÊËÌÍÎÑÒÓÔÕÖÚÜ\t䗓𨖷虴𨀐辘𧄋脟𩑢𡗶镇伢𧎼䪱轚榶𢑌㺽𤨡!#$%&(Tom@microsoft.com)*+,-./:;<=>?@[\\]^_`{|}~");
  std::vector<ustring> expect_result = ustring_vector_convertor({"aaaaaaceeeeiiinooooouu", "䗓", "𨖷", "虴", "𨀐", "辘", "𧄋", "脟", "𩑢", "𡗶", "镇", "伢", "𧎼", "䪱", "轚", "榶", "𢑌", "㺽", "𤨡", "!", "#", "$", "%", "&", "(", "tom", "@", "microsoft", ".", "com", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~"});
  BasicTokenizer tokenizer(true, true, true, true, true);
  auto result = tokenizer.Tokenize(test_case);
  EXPECT_EQ(result, expect_result);
}

TEST(tokenizer, basic_tokenizer_russia) {
  ustring test_case = ustring("A $100,000 price-tag@big>small на русском языке");
  std::vector<ustring> expect_result = ustring_vector_convertor({"a", "$", "100", ",", "000", "price", "-", "tag", "@", "big", ">", "small", "на", "русском", "языке"});
  BasicTokenizer tokenizer(true, true, true, true, true);
  auto result = tokenizer.Tokenize(test_case);
  EXPECT_EQ(result, expect_result);
}

TEST(tokenizer, basic_tokenizer) {
  ustring test_case = ustring("I mean, you’ll need something to talk about next Sunday, right?");
  std::vector<ustring> expect_result = ustring_vector_convertor({"I", "mean", ",", "you", "’", "ll", "need", "something", "to", "talk", "about", "next", "Sunday", ",", "right", "?"});
  BasicTokenizer tokenizer(false, true, true, true, true);
  auto result = tokenizer.Tokenize(test_case);
  EXPECT_EQ(result, expect_result);
}

TEST(tokenizer, truncation_one_input) {
  TruncateStrategy truncate("longest_first");

  std::vector<int64_t> init_vector1({1, 2, 3, 4, 5, 6, 7, 9});
  std::vector<int64_t> init_vector2({1, 2, 3, 4, 5});

  auto test_input = init_vector1;
  truncate.Truncate(test_input, -1);
  EXPECT_EQ(test_input, init_vector1);

  test_input = init_vector1;
  truncate.Truncate(test_input, 5);
  EXPECT_EQ(test_input, std::vector<int64_t>({1, 2, 3, 4, 5}));

  test_input = init_vector2;
  truncate.Truncate(test_input, 6);
  EXPECT_EQ(test_input, init_vector2);
}

TEST(tokenizer, truncation_longest_first) {
  TruncateStrategy truncate("longest_first");

  std::vector<int64_t> init_vector1({1, 2, 3, 4, 5, 6, 7, 9});
  std::vector<int64_t> init_vector2({1, 2, 3, 4, 5});

  auto test_input1 = init_vector1;
  auto test_input2 = init_vector2;
  truncate.Truncate(test_input1, test_input2, -1);
  EXPECT_EQ(test_input1, init_vector1);
  EXPECT_EQ(test_input2, init_vector2);

  test_input1 = init_vector1;
  test_input2 = init_vector2;
  truncate.Truncate(test_input1, test_input2, 15);
  EXPECT_EQ(test_input1, init_vector1);
  EXPECT_EQ(test_input2, init_vector2);

  test_input1 = init_vector1;
  test_input2 = init_vector2;
  truncate.Truncate(test_input1, test_input2, 14);
  EXPECT_EQ(test_input1, init_vector1);
  EXPECT_EQ(test_input2, init_vector2);

  test_input1 = init_vector1;
  test_input2 = init_vector2;
  truncate.Truncate(test_input1, test_input2, 8);
  EXPECT_EQ(test_input1, std::vector<int64_t>({1, 2, 3, 4}));
  EXPECT_EQ(test_input2, std::vector<int64_t>({1, 2, 3, 4}));

  test_input1 = init_vector1;
  test_input2 = init_vector2;
  truncate.Truncate(test_input1, test_input2, 9);
  EXPECT_EQ(test_input1, std::vector<int64_t>({1, 2, 3, 4, 5}));
  EXPECT_EQ(test_input2, std::vector<int64_t>({1, 2, 3, 4}));

  test_input1 = init_vector1;
  test_input2 = init_vector2;
  truncate.Truncate(test_input1, test_input2, 12);
  EXPECT_EQ(test_input1, std::vector<int64_t>({1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(test_input2, std::vector<int64_t>({1, 2, 3, 4, 5}));

  test_input1 = init_vector2;
  test_input2 = init_vector1;
  truncate.Truncate(test_input1, test_input2, 12);
  EXPECT_EQ(test_input1, std::vector<int64_t>({1, 2, 3, 4, 5}));
  EXPECT_EQ(test_input2, std::vector<int64_t>({1, 2, 3, 4, 5,  6 ,7}));
}