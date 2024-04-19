// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <locale>
#include "gtest/gtest.h"
#include "ocos.h"
#include "test_kernel.hpp"

#ifdef USE_CUDA

TEST(CudaOp, test_fastgelu) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(2);
  inputs[0].name = "x";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  inputs[0].dims = {6};
  inputs[0].values_float = {0., 1., 2., 3., 4., 5.};

  inputs[1].name = "bias";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  inputs[1].dims = {6};
  inputs[1].values_float = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};

  std::vector<TestValue> outputs(1);
  outputs[0].name = "y";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  outputs[0].dims = {6};
  outputs[0].values_float = {0., 0.9505811, 2.1696784, 3.298689, 4.399991, 5.5};

  std::filesystem::path model_path = "data/cuda";
  model_path /= "test_fastgelu.onnx";

  TestInference(*ort_env, model_path.c_str(), inputs, outputs);
}

#endif