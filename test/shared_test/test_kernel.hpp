// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <math.h>

const char* GetLibraryPath();

struct TestValue {
  const char* name = nullptr;
  ONNXTensorElementDataType element_type;
  std::vector<int64_t> dims;
  std::vector<float> values_float;
  std::vector<int32_t> values_int32;
  std::vector<int64_t> values_int64;
  std::vector<std::string> values_string;
  std::vector<bool> value_bool;
};

void RunSession(Ort::Session& session_object,
                const std::vector<TestValue>& inputs,
                const std::vector<TestValue>& outputs);

void TestInference(Ort::Env& env, const ORTCHAR_T* model_uri,
                   const std::vector<TestValue>& inputs,
                   const std::vector<TestValue>& outputs,
                   const char* custom_op_library_filename);
