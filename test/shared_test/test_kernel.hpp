// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <math.h>
#include "onnxruntime_cxx_api.h"

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

// output_validator is optional if you need custom validation for one or more outputs.
// for any output you do not have custom validation for call ValidateOutputEqual
using OutputValidator = std::function<void(size_t output_idx, Ort::Value& actual, TestValue expected)>;

void ValidateOutputEqual(size_t output_idx, Ort::Value& actual, TestValue expected);

void RunSession(Ort::Session& session_object,
                const std::vector<TestValue>& inputs,
                const std::vector<TestValue>& outputs,
                OutputValidator output_validator = nullptr);

void TestInference(Ort::Env& env, const ORTCHAR_T* model_uri,
                   const std::vector<TestValue>& inputs,
                   const std::vector<TestValue>& outputs,
                   OutputValidator output_validator = nullptr,
                   void* cuda_compute_stream = nullptr);

void GetTensorMutableDataString(const OrtApi& api, const OrtValue* value, std::vector<std::string>& output);
