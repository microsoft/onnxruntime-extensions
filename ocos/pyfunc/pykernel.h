// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <map>
#include "ocos.h"

struct PyCustomOpDefAttribute {
  std::string name;
  std::string dtype;
  std::string desc;
  PyCustomOpDefAttribute(const char* name, const char* dtype, const char* desc) {
    this->name = name;
    this->dtype = dtype;
    this->desc = desc;
  }
};

struct PyCustomOpDef {
  std::string op_type;
  uint64_t obj_id;
  std::vector<int> input_types;
  std::vector<int> output_types;
  std::vector<PyCustomOpDefAttribute> atts;

  static void AddOp(const PyCustomOpDef* cod);

  static const int undefined = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  static const int dt_float = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;    // maps to c type float
  static const int dt_uint8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;    // maps to c type uint8_t
  static const int dt_int8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;      // maps to c type int8_t
  static const int dt_uint16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;  // maps to c type uint16_t
  static const int dt_int16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;    // maps to c type int16_t
  static const int dt_int32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;    // maps to c type int32_t
  static const int dt_int64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;    // maps to c type int64_t
  static const int dt_string = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;  // maps to c++ type std::string
  static const int dt_bool = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  static const int dt_float16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  static const int dt_double = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;          // maps to c type double
  static const int dt_uint32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;          // maps to c type uint32_t
  static const int dt_uint64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;          // maps to c type uint64_t
  static const int dt_complex64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;    // complex with float32 real and imaginary components
  static const int dt_complex128 = ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;  // complex with float64 real and imaginary components
  static const int dt_bfloat16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;      // Non-IEEE floating-point format based on IEEE754 single-precision
};

struct PyCustomOpKernel {
  PyCustomOpKernel(OrtApi api, uint64_t id)
      : api_(api),
        ort_(api_),
        obj_id_(id) {
  }

  void Compute(OrtKernelContext* context);

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
  uint64_t obj_id_;
};

struct PyCustomOpFactory : Ort::CustomOpBase<PyCustomOpFactory, PyCustomOpKernel> {
  PyCustomOpFactory(const PyCustomOpDef* opdef) {
    if (opdef == nullptr)
      throw std::runtime_error("Python definition is empty.");
    opdef_ = opdef;
  }

  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    return new PyCustomOpKernel(api, opdef_->obj_id);
  };

  const char* GetName() const {
    return opdef_->op_type.c_str();
  };

  size_t GetInputTypeCount() const {
    return opdef_->input_types.size();
  };

  ONNXTensorElementDataType GetInputType(size_t idx) const {
    return static_cast<ONNXTensorElementDataType>(opdef_->input_types[idx]);
  };

  size_t GetOutputTypeCount() const {
    return opdef_->output_types.size();
  };

  ONNXTensorElementDataType GetOutputType(size_t idx) const {
    return static_cast<ONNXTensorElementDataType>(opdef_->output_types[idx]);
  }

  const PyCustomOpDef* opdef_;
};

std::vector<PyCustomOpFactory>& PyCustomOpDef_python_operator_list();
const PyCustomOpFactory* PyCustomOpDef_FetchPyCustomOps(size_t count);
