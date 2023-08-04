// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include "gtest/gtest.h"
#include "ocos.h"
#include "test_kernel.hpp"
#include "text/string_lower.hpp"

TEST(string_operator, test_string_lower) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(1);
  inputs[0].name = "input_1";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {3, 1};
  inputs[0].values_string = {"Abc", "Abcé", "中文"};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "customout";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = inputs[0].dims;
  outputs[0].values_string = {"abc", "abcé", "中文"};

  std::filesystem::path model_path = "data";
  model_path /= "custom_op_string_lower.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

#ifdef ENABLE_RE2_REGEX
TEST(string_operator, test_regex_split_with_offsets) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(1);
  inputs[0].name = "input:0";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {2};
  inputs[0].values_string = {"a Test 1 2 3 ♠♣", "Hi there test test ♥♦"};

  std::vector<TestValue> outputs(4);
  outputs[0].name = "output:0";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {11};
  outputs[0].values_string = {"a", "Test", "1", "2", "3", "♠♣", "Hi", "there", "test", "test", "♥♦"};

  outputs[1].name = "output1:0";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {11};
  outputs[1].values_int64 = {0, 2, 7, 9, 11, 13, 0, 3, 9, 14, 19};

  outputs[2].name = "output2:0";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {11};
  outputs[2].values_int64 = {1, 6, 8, 10, 12, 19, 2, 8, 13, 18, 25};

  outputs[3].name = "output3:0";
  outputs[3].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[3].dims = {3};
  outputs[3].values_int64 = {0, 6, 11};

  std::filesystem::path model_path = "data";
  model_path /= "test_regex_split_with_offsets.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}
#endif

TEST(string_operator, test_string_ecmaregex_replace) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(3);
  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {"a Test 1 2 3 ♠♣"};

  inputs[1].name = "pattern";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {"(\\d)"};

  inputs[2].name = "rewrite";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[2].dims = {1};
  inputs[2].values_string = {"$010"};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "output";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {"a Test 10 20 30 ♠♣"};

  std::filesystem::path model_path = "data";
  model_path /= "test_string_ecmaregex_replace.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {"a Test 10 20 30 ♠♣"};

  inputs[1].name = "pattern";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {"(\\d)"};

  inputs[2].name = "rewrite";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[2].dims = {1};
  inputs[2].values_string = {"$010"};

  outputs[0].name = "output";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {"a Test 1000 2000 3000 ♠♣"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {"a Test 10 20 30 ♠♣"};

  inputs[1].name = "pattern";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {"(\\d+)"};

  inputs[2].name = "rewrite";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[2].dims = {1};
  inputs[2].values_string = {"$010"};

  outputs[0].name = "output";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {"a Test 100 200 300 ♠♣"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {"a Test 10 20 30 ♠♣"};

  inputs[1].name = "pattern";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {"(\\w+)"};

  inputs[2].name = "rewrite";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[2].dims = {1};
  inputs[2].values_string = {"$1+"};

  outputs[0].name = "output";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {"a+ Test+ 10+ 20+ 30+ ♠♣"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {"a Test 10 20 30 ♠♣"};

  inputs[1].name = "pattern";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {"♠♣"};

  inputs[2].name = "rewrite";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[2].dims = {1};
  inputs[2].values_string = {"♣♠"};

  outputs[0].name = "output";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {"a Test 10 20 30 ♣♠"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {3};
  inputs[0].values_string = {"Test 10 20 30 ♠♣", "Test 40 50 60 🌂☂", " Test 70 80 90 🍏🍎"};

  inputs[1].name = "pattern";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {"[✀-➿🙐-🙿😀-🙏☀-⛿🌀-🗿🤀-🧿]"};

  inputs[2].name = "rewrite";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[2].dims = {1};
  inputs[2].values_string = {""};

  outputs[0].name = "output";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {3};
  outputs[0].values_string = {"Test 10 20 30 ", "Test 40 50 60 ", " Test 70 80 90 "};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {""};

  inputs[1].name = "pattern";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {"[✀-➿🙐-🙿😀-🙏☀-⛿🌀-🗿🤀-🧿]"};

  inputs[2].name = "rewrite";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[2].dims = {1};
  inputs[2].values_string = {"aa"};

  outputs[0].name = "output";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {""};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  // Test case-insensitive and non-global matching case
  model_path = model_path.parent_path();
  model_path /= "..";
  model_path /= "data";
  model_path /= "test_string_ecmaregex_replace_ignore_case_and_except_global_replace.onnx";

  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {3};
  inputs[0].values_string = {"Test test", "tEsT Test", " TEST test"};

  inputs[1].name = "pattern";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {"(test)"};

  inputs[2].name = "rewrite";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[2].dims = {1};
  inputs[2].values_string = {"$1+"};

  outputs[0].name = "output";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {3};
  outputs[0].values_string = {"Test+ test", "tEsT+ Test", " TEST+ test"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(utils, test_string_join) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(3);
  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1, 3};
  inputs[0].values_string = {"abc", "zzz", "efg"};

  inputs[1].name = "sep";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {"-"};

  inputs[2].name = "axis";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[2].dims = {1};
  inputs[2].values_int64 = {1};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "out";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {"abc-zzz-efg"};

  std::filesystem::path model_path = "data";
  model_path /= "custom_op_string_join.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(utils, test_string_join_values_empty_string) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(3);
  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {""};

  inputs[1].name = "sep";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {" "};

  inputs[2].name = "axis";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[2].dims = {1};
  inputs[2].values_int64 = {0};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "out";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {""};

  std::filesystem::path model_path = "data";
  model_path /= "custom_op_string_join.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(utils, test_string_join_dims_zero_values_empty) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(3);
  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  // If dims contains value 0, values must have length 0. See issue: https://github.com/onnx/onnx/issues/3724
  inputs[0].dims = {0};
  inputs[0].values_string = {};

  inputs[1].name = "sep";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {" "};

  inputs[2].name = "axis";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[2].dims = {1};
  inputs[2].values_int64 = {0};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "out";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {""};

  std::filesystem::path model_path = "data";
  model_path /= "custom_op_string_join.onnx";

  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(utils, test_string_join_dims_empty_values_scalar) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(3);
  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  // dims size 0 means it's scalar, and values must have 1 element. See issue: https://github.com/onnx/onnx/issues/3724
  inputs[0].dims = {};
  inputs[0].values_string = {"abc"};

  inputs[1].name = "sep";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {" "};

  inputs[2].name = "axis";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[2].dims = {1};
  inputs[2].values_int64 = {0};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "out";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {"abc"};

  std::filesystem::path model_path = "data";
  model_path /= "custom_op_string_join.onnx";

  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(string_operator, test_vector_to_string) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(1);
  inputs[0].name = "token_ids";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[0].dims = {5};
  inputs[0].values_int64 = {0, 1, 2, 3, 4};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "text";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {5};
  outputs[0].values_string = {"a", "unk", "b", "c", "unk"};

  std::filesystem::path model_path = "data";
  model_path /= "test_vector_to_string_scalar_map.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {5};
  inputs[0].values_int64 = {1000, 111, 2323, 444, 555};

  outputs[0].dims = {5};
  outputs[0].values_string = {"unk", "unk", "unk", "unk", "unk"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {0};
  inputs[0].values_int64 = {};

  outputs[0].dims = {0};
  outputs[0].values_string = {};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {5, 3};
  inputs[0].values_int64 = {0, 0, 0, 0, 0, 1, 3, 0, 1, 100, 0, 1, 43, 23, 11};

  outputs[0].dims = {5};
  outputs[0].values_string = {"a", "b", "c", "unk", "unk"};

  model_path = model_path.parent_path();
  model_path /= "..";
  model_path /= "data";
  model_path /= "test_vector_to_string_vector_map.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {0, 3};
  inputs[0].values_int64 = {};

  outputs[0].dims = {0};
  outputs[0].values_string = {};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {};
  inputs[0].values_int64 = {111};

  outputs[0].dims = {};
  outputs[0].values_string = {"unk"};

  model_path = model_path.parent_path();
  model_path /= "..";
  model_path /= "data";
  model_path /= "test_vector_to_string_scalar_input.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(string_operator, test_string_to_vector) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(1);
  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {5};
  inputs[0].values_string = {"black", "white", "black", "中文", "英文"};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "token_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {5, 3};
  outputs[0].values_int64 = {0, 1, 2, 2, 3, 4, 0, 1, 2, 3, 4, 4, -1, -1, -1};

  std::filesystem::path model_path = "data";
  model_path /= "test_string_to_vector.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {0};
  inputs[0].values_string = {};

  outputs[0].dims = {0, 3};
  outputs[0].values_int64 = {};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {1};
  inputs[0].values_string = {""};

  outputs[0].dims = {1, 3};
  outputs[0].values_int64 = {-1, -1, -1};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {};
  inputs[0].values_string = {""};

  outputs[0].dims = {3};
  outputs[0].values_int64 = {-1, -1, -1};

  model_path = model_path.parent_path();
  model_path /= "..";
  model_path /= "data";
  model_path /= "test_string_to_vector_scalar.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(string_operator, test_string_mapping) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(1);
  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {5};
  inputs[0].values_string = {"Orange and Yellow", "不知道啥颜色", "No color", "black", "white"};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "output";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {5};
  outputs[0].values_string = {"Maybe", "也不知道可不可以", "No color", "OK", "Not OK"};

  std::filesystem::path model_path = "data";
  model_path /= "test_string_mapping.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {};
  inputs[0].values_string = {"不知道啥颜色"};

  outputs[0].dims = {};
  outputs[0].values_string = {"也不知道可不可以"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {};
  inputs[0].values_string = {""};

  outputs[0].dims = {};
  outputs[0].values_string = {""};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(string_operator, test_masked_fill) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(2);
  inputs[0].name = "value";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {5};
  inputs[0].values_string = {"Orange and Yellow", "不知道啥颜色", "No color", "black", "white"};

  inputs[1].name = "mask";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  inputs[1].dims = {5};
  inputs[1].values_bool = {true, false, true, false, true};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "output";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {3};
  outputs[0].values_string = {"Orange and Yellow", "No color", "white"};

  std::filesystem::path model_path = "data";
  model_path /= "test_masked_fill.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {5};
  inputs[0].values_string = {"Orange and Yellow", "不知道啥颜色", "No color", "black", "white"};

  inputs[1].dims = {5};
  inputs[1].values_bool = {false, false, false, false, false};

  outputs[0].dims = {0};
  outputs[0].values_string = {};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {5};
  inputs[0].values_string = {"Orange and Yellow", "不知道啥颜色", "No color", "black", "white"};

  inputs[1].dims = {5};
  inputs[1].values_bool = {true, true, true, true, true};

  outputs[0].dims = {5};
  outputs[0].values_string = {"Orange and Yellow", "不知道啥颜色", "No color", "black", "white"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {1};
  inputs[0].values_string = {"a"};

  inputs[1].dims = {1};
  inputs[1].values_bool = {false};

  outputs[0].dims = {0};
  outputs[0].values_string = {};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {1};
  inputs[0].values_string = {"a"};

  inputs[1].dims = {1};
  inputs[1].values_bool = {true};

  outputs[0].dims = {1};
  outputs[0].values_string = {"a"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {0};
  inputs[0].values_string = {};

  inputs[1].dims = {0};
  inputs[1].values_bool = {};

  outputs[0].dims = {0};
  outputs[0].values_string = {};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}
