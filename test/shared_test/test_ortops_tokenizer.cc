// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include "gtest/gtest.h"
#include "ocos.h"
#include "test_kernel.hpp"

TEST(utils, test_bert_tokenizer) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(1);
  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {"We look forward to welcoming you to our stores. Whether you shop in a store or shop online, our Specialists can help you buy the products you love."};

  std::vector<TestValue> outputs(3);
  outputs[0].name = "input_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {34};
  outputs[0].values_int64 = {101, 1284, 1440, 1977, 1106, 20028, 1128, 1106, 1412, 4822, 119, 13197, 1128, 4130, 1107, 170, 2984, 1137, 4130, 3294, 117, 1412, 25607, 1116, 1169, 1494, 1128, 4417, 1103, 2982, 1128, 1567, 119, 102};

  outputs[1].name = "token_type_ids";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {34};
  outputs[1].values_int64 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  outputs[2].name = "attention_mask";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {34};
  outputs[2].values_int64 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::filesystem::path model_path = __FILE__;
  model_path = model_path.parent_path();
  model_path /= "..";
  model_path /= "data";
  model_path /= "test_bert_tokenizer1.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());


  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {"本想好好的伤感　想放任　但是没泪痕"};

  outputs[0].name = "input_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {17};
  outputs[0].values_int64 = {101, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 102};

  outputs[1].name = "token_type_ids";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {17};
  outputs[1].values_int64 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  outputs[2].name = "attention_mask";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {17};
  outputs[2].values_int64 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {"ÀÁÂÃÄÅÇÈÉÊËÌÍÎÑÒÓÔÕÖÚÜ\t䗓𨖷虴𨀐辘𧄋脟𩑢𡗶镇伢𧎼䪱轚榶𢑌㺽𤨡!#$%&(Tom@microsoft.com)*+,-./:;<=>?@[\\]^_`{|}~"};

  outputs[0].name = "input_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {71};
  outputs[0].values_int64 = {101, 13807, 11189, 8101, 27073, 27073, 12738, 11607, 2346, 2346, 2346, 2346, 2346, 2591, 2591, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 106, 108, 109, 110, 111, 113, 2545, 137, 17599, 7301, 4964, 119, 3254, 114, 115, 116, 117, 118, 119, 120, 131, 132, 133, 134, 135, 136, 137, 164, 165, 166, 167, 168, 169, 196, 197, 198, 199, 102};

  outputs[1].name = "token_type_ids";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {71};
  outputs[1].values_int64 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  outputs[2].name = "attention_mask";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {71};
  outputs[2].values_int64 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}