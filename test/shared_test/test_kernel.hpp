// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <math.h>
#include "onnxruntime_cxx_api.h"

const char* GetLibraryPath();

struct TestValue {
  TestValue(const char* name_in, const std::vector<float>& values_in, const std::vector<int64_t>& dims_in)
      : name{name_in}, element_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}, values_float{values_in}, dims{dims_in} {}

  TestValue(const char* name_in, const std::vector<uint8_t>& values_in, const std::vector<int64_t>& dims_in)
      : name{name_in}, element_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8}, values_uint8{values_in}, dims{dims_in} {}

  TestValue(const char* name_in, const std::vector<int32_t>& values_in, const std::vector<int64_t>& dims_in)
      : name{name_in}, element_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32}, values_int32{values_in}, dims{dims_in} {}

  TestValue(const char* name_in, const std::vector<int64_t>& values_in, const std::vector<int64_t>& dims_in)
      : name{name_in}, element_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64}, values_int64{values_in}, dims{dims_in} {}

  TestValue(const char* name_in, const std::vector<std::string>& values_in, const std::vector<int64_t>& dims_in)
      : name{name_in}, element_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING}, values_string{values_in}, dims{dims_in} {}

  TestValue(const char* name_in, const std::vector<bool>& values_in, const std::vector<int64_t>& dims_in)
      : name{name_in}, element_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL}, values_bool{values_in}, dims{dims_in} {}

  TestValue() = default;

  const char* name = nullptr;
  ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::vector<int64_t> dims;
  std::vector<float> values_float;
  std::vector<uint8_t> values_uint8;
  std::vector<int32_t> values_int32;
  std::vector<int64_t> values_int64;
  std::vector<std::string> values_string;
  std::vector<bool> values_bool;
};

void RunSession(Ort::Session& session_object,
                const std::vector<TestValue>& inputs,
                const std::vector<TestValue>& outputs);

void TestInference(Ort::Env& env, const ORTCHAR_T* model_uri,
                   const std::vector<TestValue>& inputs,
                   const std::vector<TestValue>& outputs,
                   const char* custom_op_library_filename);
