// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include "ocos.h"
#include "test_kernel.hpp"
#include "kernels/string_join.hpp"
#include <filesystem>

TEST(utils, test_string_join) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(3);
  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {3, 2};
  inputs[0].values_string = {"Abc", "a", "Abcé", "bb", "中文", "c"};

  inputs[1].name = "sep";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {","};

  inputs[2].name = "axis";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[2].dims = {1};
  inputs[2].values_int64 = {1};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "customout";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {3};
  outputs[0].values_string = {"Abc,a", "Abcé,bb", "中文,c"};

  std::filesystem::path model_path = __FILE__;
  model_path = model_path.parent_path();
  model_path /= "..";
  model_path /= "data";
  model_path /= "custom_op_string_join.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(utils, test_string_split) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(3);
  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {4};
  inputs[0].values_string = {"a,,b", "", "aa,b,c", "dddddd"};

  inputs[1].name = "delimiter";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[1].dims = {1};
  inputs[1].values_string = {","};

  inputs[2].name = "skip_empty";
  inputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  inputs[2].dims = {1};
  inputs[2].values_bool = {false};

  std::vector<TestValue> outputs(3);
  outputs[0].name = "indices";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {7, 2};
  outputs[0].values_int64 = {0, 0, 0, 1, 0, 2, 2, 0, 2, 1, 2, 2, 3, 0};

  outputs[1].name = "values";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[1].dims = {7};
  outputs[1].values_string = {"a", "", "b", "aa", "b", "c", "dddddd"};

  outputs[2].name = "shape";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {2};
  outputs[2].values_int64 = {4, 3};

  std::filesystem::path model_path = __FILE__;
  model_path = model_path.parent_path();
  model_path /= "..";
  model_path /= "data";
  model_path /= "custom_op_string_split.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}
