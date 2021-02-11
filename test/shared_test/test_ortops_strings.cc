// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include "ocos.h"
#include "test_kernel.hpp"
#include "kernels/string_lower.hpp"
#include <filesystem>

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
