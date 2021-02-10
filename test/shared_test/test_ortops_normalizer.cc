// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include "ocos.h"
#include "kernels/kernels.h"
#include "../tokenizer/string_normalizer.hpp"
#include "test_kernel.hpp"
#include <filesystem>

static CustomOpStringNormalizer c_CustomOpStringNormalizer;

TEST(utils, test_ort_normalizer) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
  std::cout << "Running custom op inference" << std::endl;

  std::vector<TestValue> inputs(1);
  inputs[0].name = "input_1";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1, 1};
  inputs[0].values_string = {"Abc"};

  // prepare expected inputs and outputs
  std::vector<TestValue> outputs(1);
  outputs[0].name = "output_1";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1, 1};
  outputs[0].values_string = {"Abc"};

  std::filesystem::path model_path = __FILE__;
  model_path = model_path.parent_path();
  model_path /= "..";
  model_path /= "data";
  model_path /= "custom_op_string_normalizer.onnx";
  AddExternalCustomOp(&c_CustomOpStringNormalizer);
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}
