// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <locale>
#include "gtest/gtest.h"
#include "ocos.h"
#include "test_kernel.hpp"

void TestHfBertTokenizerHelper(const std::filesystem::path& test_model_path,
                               bool validate_optional_token_type_ids_output) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  const std::vector<int64_t> input_dims{2};
  const std::string context1 =
      "John is a 10 year old boy. "
      "He is the son of Robert Smith. "
      "Elizabeth Davis is Robert's wife. "
      "She teaches at UC Berkeley. "
      "Sophia Smith is Elizabeth's daughter. "
      "She studies at UC Davis";
  const std::string context2 =
      "My name is John."
      "I live in San Jose, California."
      "Rob is my friend."
      "He lives in Seattle, Washington.";

  std::vector<std::vector<TestValue>> inputs = {
      {TestValue{"text", {"Who is John's sister?", context1}, input_dims}},
      {TestValue{"text", {"Where does sophia study?", context1}, input_dims}},
      {TestValue{"text", {"Who is John's mom?", context1}, input_dims}},
      {TestValue{"text", {"Where does John's father's wife teach?", context1}, input_dims}},
      {TestValue{"text", {"Who is John's father's wife's daughter's brother?", context1}, input_dims}},
      {TestValue{"text", {"Who is John's friend?", context2}, input_dims}},
      {TestValue{"text", {"Where does John's friend live?", context2}, input_dims}},
      {TestValue{"text", {"Which state does John's friend live?", context2}, input_dims}}};

  std::vector<std::vector<TestValue>> outputs = {
      {TestValue{"input_ids",
                 std::vector<int64_t>{101, 2040, 2003, 2198, 1005, 1055, 2905, 1029, 102, 2198, 2003, 1037, 2184, 2095, 2214,
                                      2879, 1012, 2002, 2003, 1996, 2365, 1997, 2728, 3044, 1012, 3870, 4482, 2003, 2728,
                                      1005, 1055, 2564, 1012, 2016, 12011, 2012, 15384, 8256, 1012, 9665, 3044, 2003, 3870,
                                      1005, 1055, 2684, 1012, 2016, 2913, 2012, 15384, 4482, 102},
                 {1, 53}},
       TestValue{"attention_mask", std::vector<int64_t>(53, 1), {1, 53}}},

      {TestValue{"input_ids",
                 std::vector<int64_t>{101, 2073, 2515, 9665, 2817, 1029, 102, 2198, 2003, 1037, 2184, 2095, 2214, 2879,
                                      1012, 2002, 2003, 1996, 2365, 1997, 2728, 3044, 1012, 3870, 4482, 2003, 2728,
                                      1005, 1055, 2564, 1012, 2016, 12011, 2012, 15384, 8256, 1012, 9665, 3044, 2003,
                                      3870, 1005, 1055, 2684, 1012, 2016, 2913, 2012, 15384, 4482, 102},
                 {1, 51}},
       TestValue{"attention_mask", std::vector<int64_t>(51, 1), {1, 51}}},

      {TestValue{"input_ids",
                 std::vector<int64_t>{101, 2040, 2003, 2198, 1005, 1055, 3566, 1029, 102, 2198, 2003, 1037, 2184, 2095,
                                      2214, 2879, 1012, 2002, 2003, 1996, 2365, 1997, 2728, 3044, 1012, 3870, 4482,
                                      2003, 2728, 1005, 1055, 2564, 1012, 2016, 12011, 2012, 15384, 8256, 1012, 9665,
                                      3044, 2003, 3870, 1005, 1055, 2684, 1012, 2016, 2913, 2012, 15384, 4482, 102},
                 {1, 53}},
       TestValue{"attention_mask", std::vector<int64_t>(53, 1), {1, 53}}},

      {TestValue{"input_ids",
                 std::vector<int64_t>{101, 2073, 2515, 2198, 1005, 1055, 2269, 1005, 1055, 2564, 6570, 1029, 102, 2198,
                                      2003, 1037, 2184, 2095, 2214, 2879, 1012, 2002, 2003, 1996, 2365, 1997, 2728,
                                      3044, 1012, 3870, 4482, 2003, 2728, 1005, 1055, 2564, 1012, 2016, 12011, 2012,
                                      15384, 8256, 1012, 9665, 3044, 2003, 3870, 1005, 1055, 2684, 1012, 2016, 2913,
                                      2012, 15384, 4482, 102},
                 {1, 57}},
       TestValue{"attention_mask", std::vector<int64_t>(57, 1), {1, 57}}},

      {TestValue{"input_ids",
                 std::vector<int64_t>{101, 2040, 2003, 2198, 1005, 1055, 2269, 1005, 1055, 2564, 1005, 1055, 2684,
                                      1005, 1055, 2567, 1029, 102, 2198, 2003, 1037, 2184, 2095, 2214, 2879, 1012,
                                      2002, 2003, 1996, 2365, 1997, 2728, 3044, 1012, 3870, 4482, 2003, 2728, 1005,
                                      1055, 2564, 1012, 2016, 12011, 2012, 15384, 8256, 1012, 9665, 3044, 2003, 3870,
                                      1005, 1055, 2684, 1012, 2016, 2913, 2012, 15384, 4482, 102},
                 {1, 62}},
       TestValue{"attention_mask", std::vector<int64_t>(62, 1), {1, 62}}},

      {TestValue{"input_ids",
                 std::vector<int64_t>{101, 2040, 2003, 2198, 1005, 1055, 2767, 1029, 102, 2026, 2171, 2003, 2198, 1012,
                                      1045, 2444, 1999, 2624, 4560, 1010, 2662, 1012, 6487, 2003, 2026, 2767, 1012,
                                      2002, 3268, 1999, 5862, 1010, 2899, 1012, 102},
                 {1, 35}},
       TestValue{"attention_mask", std::vector<int64_t>(35, 1), {1, 35}}},

      {TestValue{"input_ids",
                 std::vector<int64_t>{101, 2073, 2515, 2198, 1005, 1055, 2767, 2444, 1029, 102, 2026, 2171, 2003, 2198,
                                      1012, 1045, 2444, 1999, 2624, 4560, 1010, 2662, 1012, 6487, 2003, 2026, 2767,
                                      1012, 2002, 3268, 1999, 5862, 1010, 2899, 1012, 102},
                 {1, 36}},
       TestValue{"attention_mask", std::vector<int64_t>(36, 1), {1, 36}}},

      {TestValue{"input_ids",
                 std::vector<int64_t>{101, 2029, 2110, 2515, 2198, 1005, 1055, 2767, 2444, 1029, 102, 2026, 2171, 2003,
                                      2198, 1012, 1045, 2444, 1999, 2624, 4560, 1010, 2662, 1012, 6487, 2003, 2026,
                                      2767, 1012, 2002, 3268, 1999, 5862, 1010, 2899, 1012, 102},
                 {1, 37}},
       TestValue{"attention_mask", std::vector<int64_t>(37, 1LL), {1, 37}}}};

  if (validate_optional_token_type_ids_output) {
    outputs[0].push_back(TestValue{"token_type_ids",
                                   std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                   {1, 53}});
    outputs[1].push_back(TestValue{"token_type_ids",
                                   std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                   {1, 51}});
    outputs[2].push_back(TestValue{"token_type_ids",
                                   std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                   {1, 53}});
    outputs[3].push_back(TestValue{"token_type_ids",
                                   std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        1, 1, 1},
                                   {1, 57}});
    outputs[4].push_back(TestValue{"token_type_ids",
                                   std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        1, 1, 1, 1, 1, 1, 1, 1},
                                   {1, 62}});
    outputs[5].push_back(TestValue{"token_type_ids",
                                   std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        1, 1, 1, 1, 1, 1, 1, 1},
                                   {1, 35}});
    outputs[6].push_back(TestValue{"token_type_ids",
                                   std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        1, 1, 1, 1, 1, 1, 1, 1, 1},
                                   {1, 36}});
    outputs[7].push_back(TestValue{"token_type_ids",
                                   std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                   {1, 37}});
  }

  ASSERT_EQ(inputs.size(), outputs.size());

  for (size_t i = 0, count = inputs.size(); i < count; ++i) {
    TestInference(*ort_env, test_model_path.c_str(), inputs[i], outputs[i]);
  }
}

TEST(hf_bert_tokenizer_operator, test_default) {
  const std::filesystem::path model_path = "data/bert-large-uncased-whole-word-masking-finetuned-squad-tokenizer.onnx";
  TestHfBertTokenizerHelper(model_path, true);
}

TEST(hf_bert_tokenizer_operator, test_optional_output) {
  const std::filesystem::path model_path = "data/test_hf_bert_tokenizer_optional_output.onnx";
  TestHfBertTokenizerHelper(model_path, false);
}
