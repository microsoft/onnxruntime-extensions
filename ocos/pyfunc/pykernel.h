// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <map>

#include "ocos.h"

struct PyCustomOpDef {
  std::string op_type;
  uint64_t obj_id;
  std::vector<int> input_types;
  std::vector<int> output_types;

  static std::vector<const PyCustomOpDef*>& FullList() {
    static std::vector<const PyCustomOpDef*> lst_custom_opdef;
    return lst_custom_opdef;
  }

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

  static const std::map<int, int>& get_numpy_type_map(bool from_or_to);
};

struct PyCustomOpKernel {
  PyCustomOpKernel(OrtApi api)
      : api_(api),
        ort_(api_),
        obj_id_(0) {
  }

  void Compute(OrtKernelContext* context);
  void set_opdef_id(uint64_t id) {
    obj_id_ = id;
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
  uint64_t obj_id_;
};

struct PyCustomOpFactory : Ort::CustomOpBase<PyCustomOpFactory, PyCustomOpKernel> {
  PyCustomOpFactory(PyCustomOpDef const* opdef) {
    opdef_ = opdef;
  }

  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    auto kernel = new PyCustomOpKernel(api);
    kernel->set_opdef_id(opdef_ == nullptr? uint64_t(0):opdef_->obj_id);
    return kernel;
  };

  const char* GetName() const {
    return opdef_ == nullptr ? "Unknown" : opdef_->op_type.c_str();
  };

  size_t GetInputTypeCount() const {
    return opdef_ == nullptr ? 1 : opdef_->input_types.size();
  };

  ONNXTensorElementDataType GetInputType(size_t idx) const {
    return opdef_ == nullptr ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : static_cast<ONNXTensorElementDataType>(opdef_->input_types[idx]);
  };

  size_t GetOutputTypeCount() const {
    return opdef_ == nullptr ? 1 : opdef_->output_types.size();
  };

  ONNXTensorElementDataType GetOutputType(size_t idx) const {
    return opdef_ == nullptr ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : static_cast<ONNXTensorElementDataType>(opdef_->output_types[idx]);
  };

  PyCustomOpDef const* opdef_ = nullptr;
};
