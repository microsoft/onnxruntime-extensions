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

  static void AddOp(const PyCustomOpDef* cod);

  // no initializer here to avoid gcc whole-archive
  static const int undefined;
  static const int dt_float;
  static const int dt_uint8;
  static const int dt_int8;
  static const int dt_uint16;
  static const int dt_int16;
  static const int dt_int32;
  static const int dt_int64;
  static const int dt_string;
  static const int dt_bool;
  static const int dt_float16;
  static const int dt_double;
  static const int dt_uint32;
  static const int dt_uint64;
  static const int dt_complex64;
  static const int dt_complex128;
  static const int dt_bfloat16;
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
