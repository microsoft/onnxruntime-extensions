// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include "gtest/gtest.h"
#include "ocos.h"
#include "test_kernel.hpp"
#include "text/string_lower.hpp"


TEST(utils, test_string_lower) {
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

  std::filesystem::path model_path = __FILE__;
  model_path = model_path.parent_path();
  model_path /= "..";
  model_path /= "data";
  model_path /= "custom_op_string_lower.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}


TEST(utils, test_regex_split_with_offsets) {
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

  std::filesystem::path model_path = __FILE__;
  model_path = model_path.parent_path();
  model_path /= "..";
  model_path /= "data";
  model_path /= "test_regex_split_with_offsets.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}
