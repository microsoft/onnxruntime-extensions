// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include "test_utils.h"
#include "ocos.h"

#include "cc_test/test_kernel.hpp"


struct Input {
  const char* name = nullptr;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

void RunSession(Ort::Session& session_object,
                const std::vector<Input>& inputs,
                const char* output_name,
                const std::vector<int64_t>& dims_y,
                const std::vector<int32_t>& values_y) {

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info, 
      const_cast<float*>(inputs[i].values.data()), inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }

  std::vector<Ort::Value> ort_outputs;
  ort_outputs = session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(), &output_name, 1);
  ASSERT_EQ(ort_outputs.size(), 1u);
  auto output_tensor = &ort_outputs[0];

  auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), dims_y);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_y.size(), total_len);

  int32_t* f = output_tensor->GetTensorMutableData<int32_t>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }
}

void TestInference(Ort::Env& env, const ORTCHAR_T* model_uri,
                   const std::vector<Input>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<int32_t>& expected_values_y,
                   const char* custom_op_library_filename) {
  Ort::SessionOptions session_options;
  void* handle = nullptr;
  if (custom_op_library_filename) {
    Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary((OrtSessionOptions*)session_options, custom_op_library_filename, &handle));
  }

  // if session creation passes, model loads fine
  Ort::Session session(env, model_uri, session_options);

  // Now run
  RunSession(session,
              inputs,
              output_name,
              expected_dims_y,
              expected_values_y);
}

static CustomOpOne op_1st;
static CustomOpTwo op_2nd;

TEST(utils, test_ort_case) {
  
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
  std::cout << "Running custom op inference" << std::endl;

  std::vector<Input> inputs(2);
  inputs[0].name = "input_1";
  inputs[0].dims = {3, 5};
  inputs[0].values = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                      6.6f, 7.7f, 8.8f, 9.9f, 10.0f,
                      11.1f, 12.2f, 13.3f, 14.4f, 15.5f};
  inputs[1].name = "input_2";
  inputs[1].dims = {3, 5};
  inputs[1].values = {15.5f, 14.4f, 13.3f, 12.2f, 11.1f,
                      10.0f, 9.9f, 8.8f, 7.7f, 6.6f,
                      5.5f, 4.4f, 3.3f, 2.2f, 1.1f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 5};
  std::vector<int32_t> expected_values_y =
      {17, 17, 17, 17, 17,
       17, 18, 18, 18, 17,
       17, 17, 17, 17, 17};

#if defined(_WIN32)
  const char lib_name[] = "ortcustomops.dll";
  const ORTCHAR_T model_path[] = L"test\\data\\custom_op_test.onnx";
#elif defined(__APPLE__)
  const char lib_name[] = "libortcustomops.dylib";
  const ORTCHAR_T model_path[] = "test/data/custom_op_test.onnx";
#else
  const char lib_name[] = "./libortcustomops.so";
  const ORTCHAR_T model_path[] = "test/data/custom_op_test.onnx";
#endif
  AddExternalCustomOp(&op_1st);
  AddExternalCustomOp(&op_2nd);
  TestInference(*ort_env, model_path, inputs, "output", expected_dims_y, expected_values_y, lib_name);
}
