// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <locale>
#include "gtest/gtest.h"
#include "ocos.h"
#include "test_kernel.hpp"

TEST(tokenizer_opertors, test_bert_tokenizer) {
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

  std::filesystem::path model_path = "data";
  model_path /= "test_bert_tokenizer.onnx";
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

  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {""};

  outputs[0].name = "input_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {2};
  outputs[0].values_int64 = {101, 102};

  outputs[1].name = "token_type_ids";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {2};
  outputs[1].values_int64 = {0, 0};

  outputs[2].name = "attention_mask";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {2};
  outputs[2].values_int64 = {1, 1};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {2};
  inputs[0].values_string = {"M1 Pro and M1 Max scale the amazing M1 architecture to new heights — and for the first time, they bring a system on a chip (SoC) architecture to a pro notebook.",
                             "Both have more CPU cores, more GPU cores, and more unified memory than M1. Along with a powerful Neural Engine for supercharged machine learning and upgraded media engines with ProRes support, M1 Pro and M1 Max allow pros to do things they never could before."};

  outputs[0].name = "input_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {99};
  outputs[0].values_int64 = {101, 26528, 5096, 1105, 26528, 3405, 3418, 1103, 6929, 26528, 4220, 1106, 1207, 16291, 100, 1105, 1111, 1103, 1148, 1159, 117, 1152, 2498, 170, 1449, 1113, 170, 11451, 113, 1573, 1658, 114, 4220, 1106, 170, 5250, 17189, 119, 102, 2695, 1138, 1167, 18701, 4160, 1116, 117, 1167, 15175, 2591, 4160, 1116, 117, 1105, 1167, 13943, 2962, 1190, 26528, 119, 6364, 1114, 170, 3110, 151, 8816, 1348, 13451, 1111, 7688, 23131, 3395, 3776, 1105, 9554, 2394, 4540, 1114, 5096, 2069, 1279, 1619, 117, 26528, 5096, 1105, 26528, 3405, 2621, 5250, 1116, 1106, 1202, 1614, 1152, 1309, 1180, 1196, 119, 102};

  outputs[1].name = "token_type_ids";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {99};
  outputs[1].values_int64 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  outputs[2].name = "attention_mask";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {99};
  outputs[2].values_int64 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {2};
  inputs[0].values_string = {"a", ""};

  outputs[0].name = "input_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {4};
  outputs[0].values_int64 = {101, 170, 102, 102};

  outputs[1].name = "token_type_ids";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {4};
  outputs[1].values_int64 = {0, 0, 0, 1};

  outputs[2].name = "attention_mask";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {4};
  outputs[2].values_int64 = {1, 1, 1, 1};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {1};
  inputs[0].values_string = {""};

  outputs[0].name = "input_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {2};
  outputs[0].values_int64 = {101, 102};

  outputs[1].name = "token_type_ids";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {2};
  outputs[1].values_int64 = {0, 0};

  outputs[2].name = "attention_mask";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {2};
  outputs[2].values_int64 = {1, 1};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {2};
  inputs[0].values_string = {"M1 Pro and M1 Max scale the amazing M1 architecture to new heights — and for the first time, they bring a system on a chip (SoC) architecture to a pro notebook.",
                             "Both have more CPU cores, more GPU cores, and more unified memory than M1. Along with a powerful Neural Engine for supercharged machine learning and upgraded media engines with ProRes support, M1 Pro and M1 Max allow pros to do things they never could before."};

  outputs[0].name = "input_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {99};
  outputs[0].values_int64 = {101, 26528, 5096, 1105, 26528, 3405, 3418, 1103, 6929, 26528, 4220, 1106, 1207, 16291, 100, 1105, 1111, 1103, 1148, 1159, 117, 1152, 2498, 170, 1449, 1113, 170, 11451, 113, 1573, 1658, 114, 4220, 1106, 170, 5250, 17189, 119, 102, 2695, 1138, 1167, 18701, 4160, 1116, 117, 1167, 15175, 2591, 4160, 1116, 117, 1105, 1167, 13943, 2962, 1190, 26528, 119, 6364, 1114, 170, 3110, 151, 8816, 1348, 13451, 1111, 7688, 23131, 3395, 3776, 1105, 9554, 2394, 4540, 1114, 5096, 2069, 1279, 1619, 117, 26528, 5096, 1105, 26528, 3405, 2621, 5250, 1116, 1106, 1202, 1614, 1152, 1309, 1180, 1196, 119, 102};

  outputs[1].name = "token_type_ids";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {99};
  outputs[1].values_int64 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  outputs[2].name = "attention_mask";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {99};
  outputs[2].values_int64 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {2};
  inputs[0].values_string = {"", "a"};

  outputs[0].name = "input_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {4};
  outputs[0].values_int64 = {101, 102, 170, 102};

  outputs[1].name = "token_type_ids";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {4};
  outputs[1].values_int64 = {0, 0, 1, 1};

  outputs[2].name = "attention_mask";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {4};
  outputs[2].values_int64 = {1, 1, 1, 1};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(tokenizer_opertors, test_bert_tokenizer_scalar) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(1);
  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {};
  inputs[0].values_string = {"We look forward to welcoming you to our stores. Whether you shop in a store or shop online, our Specialists can help you buy the products you love."};

  std::vector<TestValue> outputs(3);
  outputs[0].name = "input_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {33};
  outputs[0].values_int64 = {101, 1195, 1440, 1977, 1106, 20028, 1128, 1106, 1412, 4822, 119, 2480, 1128, 4130, 1107, 170, 2984, 1137, 4130, 3294, 117, 1412, 18137, 1169, 1494, 1128, 4417, 1103, 2982, 1128, 1567, 119, 102};

  outputs[1].name = "token_type_ids";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {33};
  outputs[1].values_int64 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  outputs[2].name = "attention_mask";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {33};
  outputs[2].values_int64 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::filesystem::path model_path = "data";
  model_path /= "test_bert_tokenizer_scalar.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  // change locale to system locale
  std::locale();

  inputs[0].name = "text";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {};
  inputs[0].values_string = {
      "再见我的爱\n"
      "I wanna say goodbye\n"
      "再见我的过去\n"
      "I want a new life"};

  outputs[0].name = "input_ids";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {22};
  outputs[0].values_int64 = {101, 100, 100, 100, 100, 100, 178, 16445, 1474, 12903, 100, 100, 100, 100, 100, 100, 178, 1328, 170, 1207, 1297, 102};

  outputs[1].name = "token_type_ids";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {22};
  outputs[1].values_int64 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  outputs[2].name = "attention_mask";
  outputs[2].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[2].dims = {22};
  outputs[2].values_int64 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(tokenizer_opertors, test_bert_tokenizer_decoder) {

  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(2);
  inputs[0].name = "ids";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[0].dims = {1};
  inputs[0].values_int64 = {11};

  inputs[1].name = "position";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[1].dims = {0,2};
  inputs[1].values_int64 = {};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "str";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {"[unused11]"};

  std::filesystem::path model_path = "data";
  model_path /= "test_bert_tokenizer_decoder_without_indices.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {5};
  inputs[0].values_int64 = {101, 2774, 102, 2774, 102};

  inputs[1].dims = {0,2};
  inputs[1].values_int64 = {};

  outputs[0].dims = {1};
  outputs[0].values_string = {"[CLS] test [SEP] test [SEP]"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {31};
  inputs[0].values_int64 = {101, 164, 17599, 7301, 4964, 120, 1113, 21123, 10607, 4974, 118, 16003, 166, 2508, 12272, 1514, 3392, 2607, 1154, 3392, 1231, 1233, 118, 121, 119, 125, 113, 11629, 108, 20977, 102};

  inputs[1].dims = {0,2};
  inputs[1].values_int64 = {};

  outputs[0].dims = {1};
  outputs[0].values_string = {"[CLS] [microsoft/onnxruntime-extensions] Merge main branch changes into branch rel-0. 4 (PR # 178 [SEP]"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {5};
  inputs[0].values_int64 = {21123, 10607, 4974, 118, 16003};

  inputs[1].dims = {0,2};
  inputs[1].values_int64 = {};

  outputs[0].dims = {1};
  outputs[0].values_string = {"-extensions"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {28};
  inputs[0].values_int64 = {163, 4638, 4538, 19009, 1708, 27516, 6592, 12324, 2137, 1161, 2137, 1162, 2101, 1394, 3663, 1158, 117, 23816, 2162, 3814, 1658, 10654, 1182, 1708, 9435, 7231, 1658, 6530};

  inputs[1].dims = {0,2};
  inputs[1].values_int64 = {};

  outputs[0].dims = {1};
  outputs[0].values_string = {"ZheJiuShisuibianDaDePinYing, YongLaiCeshiSuffixCase"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {2};
  inputs[0].values_int64 = {-1, 28997};

  inputs[1].dims = {0,2};
  inputs[1].values_int64 = {};

  outputs[0].dims = {1};
  outputs[0].values_string = {"[UNK] [UNK]"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

TEST(tokenizer_opertors, test_bert_tokenizer_decoder_with_idices) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(2);
  inputs[0].name = "ids";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[0].dims = {1};
  inputs[0].values_int64 = {11};

  inputs[1].name = "position";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[1].dims = {1,2};
  inputs[1].values_int64 = {0, 1};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "str";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  outputs[0].dims = {1};
  outputs[0].values_string = {"[unused11]"};

  std::filesystem::path model_path = "data";
  model_path /= "test_bert_tokenizer_decoder_with_indices.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {30};
  inputs[0].values_int64 = {101, 163, 4638, 4538, 19009, 1708, 27516, 6592, 12324, 2137, 1161, 2137, 1162, 2101, 1394, 3663, 1158, 117, 23816, 2162, 3814, 1658, 10654, 1182, 1708, 9435, 7231, 1658, 6530, 102};

  inputs[1].dims = {0,2};
  inputs[1].values_int64 = {};

  outputs[0].dims = {0};
  outputs[0].values_string = {};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());


  inputs[0].dims = {30};
  inputs[0].values_int64 = {101, 163, 4638, 4538, 19009, 1708, 27516, 6592, 12324, 2137, 1161, 2137, 1162, 2101, 1394, 3663, 1158, 117, 23816, 2162, 3814, 1658, 10654, 1182, 1708, 9435, 7231, 1658, 6530, 102};

  inputs[1].dims = {4,2};
  inputs[1].values_int64 = {0, 3, 5, 10, 17, 18, 18, 30};

  outputs[0].dims = {4};
  outputs[0].values_string = {"Zhe", "", ",", "YongLaiCeshiSuffixCase"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  inputs[0].dims = {30};
  inputs[0].values_int64 = {101, 163, 4638, 4538, 19009, 1708, 27516, 6592, 12324, 2137, 1161, 2137, 1162, 2101, 1394, 3663, 1158, 117, 23816, 2162, 3814, 1658, 10654, 1182, 1708, 9435, 7231, 1658, 6530, 102};

  inputs[1].dims = {1,2};
  inputs[1].values_int64 = {0, 30};

  outputs[0].dims = {1};
  outputs[0].values_string = {"ZheJiuShisuibianDaDePinYing, YongLaiCeshiSuffixCase"};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}