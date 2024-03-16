// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include "gtest/gtest.h"
#include "ocos.h"
#include "test_kernel.hpp"

#include "operators/math/negpos.hpp"

TEST(math_operator, eager_poc){
  auto test_allocator = std::make_unique<ortc::TestAllocator>();
  std::vector<float> input_data = {0.0f, 0.2f, -1.3f, 1.5f};

  ortc::Tensor<float> input(std::vector<int64_t>{2, 2}, input_data.data());

  ortc::Tensor<float> output1(test_allocator.get());
  ortc::Tensor<float> output2(test_allocator.get());

  auto result = neg_pos(input, output1, output2);
  assert(!result);
  assert(output1.Shape() == input.Shape() && output2.Shape() == input.Shape());
}

TEST(math_operator, segment_extraction) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(1);
  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[0].dims = {1, 11};
  inputs[0].values_int64 = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3};

  std::vector<TestValue> outputs(2);
  outputs[0].name = "position";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {3,2};
  outputs[0].values_int64 = {2, 4, 4, 7, 7, 11};

  outputs[1].name = "value";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {3};
  outputs[1].values_int64 = {1, 2, 3};

  std::filesystem::path model_path = "data";
  model_path /= "test_segment_extraction.onnx";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs);

  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[0].dims = {1, 12};
  inputs[0].values_int64 = {1, 1, 0, 0, 2, 2, 2, 3, 3, 3, 0, 5};

  outputs[0].name = "position";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {4,2};
  outputs[0].values_int64 = {0, 2, 4, 7, 7, 10, 11, 12};

  outputs[1].name = "value";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {4};
  outputs[1].values_int64 = {1, 2, 3, 5};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs);


  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[0].dims = {1, 4};
  inputs[0].values_int64 = {1, 2, 4, 5};

  outputs[0].name = "position";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {4,2};
  outputs[0].values_int64 = {0, 1, 1, 2, 2, 3, 3, 4};

  outputs[1].name = "value";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {4};
  outputs[1].values_int64 = {1, 2, 4, 5};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs);


  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[0].dims = {1, 13};
  inputs[0].values_int64 = {0, 0, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 0};

  outputs[0].name = "position";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {2,2};
  outputs[0].values_int64 = {2, 5, 9, 12};

  outputs[1].name = "value";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {2};
  outputs[1].values_int64 = {1, 3};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs);

  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[0].dims = {1, 1};
  inputs[0].values_int64 = {0};

  outputs[0].name = "position";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {0, 2};
  outputs[0].values_int64 = {};

  outputs[1].name = "value";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {0};
  outputs[1].values_int64 = {};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs);

  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[0].dims = {1, 0};
  inputs[0].values_int64 = {0};

  outputs[0].name = "position";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {0, 2};
  outputs[0].values_int64 = {};

  outputs[1].name = "value";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {0};
  outputs[1].values_int64 = {};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs);


  inputs[0].name = "input";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  inputs[0].dims = {1, 1};
  inputs[0].values_int64 = {1};

  outputs[0].name = "position";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {1, 2};
  outputs[0].values_int64 = {0, 1};

  outputs[1].name = "value";
  outputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[1].dims = {1};
  outputs[1].values_int64 = {1};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs);
}