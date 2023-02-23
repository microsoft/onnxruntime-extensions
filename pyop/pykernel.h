// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <map>
#include "ocos.h"

struct PyCustomOpDef {
  std::string op_type;
  uint64_t obj_id = 0;
  std::vector<int> input_types;
  std::vector<int> output_types;
  std::vector<std::string> attrs;

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
  PyCustomOpKernel(const OrtApi& api, const OrtKernelInfo* info, uint64_t id, const std::vector<std::string>& attrs);
  void Compute(OrtKernelContext* context);

 private:
  const OrtApi& api_;
  OrtW::CustomOpApi ort_;
  uint64_t obj_id_;
  std::map<std::string, std::string> attrs_values_;
};

struct PyCustomOpFactory : OrtW::CustomOpBase<PyCustomOpFactory, PyCustomOpKernel> {
  PyCustomOpFactory() {
    // STL vector needs it.
  }

  PyCustomOpFactory(const PyCustomOpDef* opdef, const std::string& domain, const std::string& op) {
    if (opdef == nullptr)
      throw std::runtime_error("Python definition is empty.");
    opdef_ = opdef;
    op_domain_ = domain;
    op_type_ = op;
  }

  const char* GetName() const {
    return op_type_.c_str();
  };

  size_t GetInputTypeCount() const {
    return opdef_->input_types.size();
  };

  ONNXTensorElementDataType GetInputType(size_t idx) const {
    return static_cast<ONNXTensorElementDataType>(opdef_->input_types[idx]);
  };

  const std::vector<std::string>& GetAttributesNames() const {
    return opdef_->attrs;
  }

  size_t GetOutputTypeCount() const {
    return opdef_->output_types.size();
  };

  ONNXTensorElementDataType GetOutputType(size_t idx) const {
    return static_cast<ONNXTensorElementDataType>(opdef_->output_types[idx]);
  }

  const PyCustomOpDef* opdef_ = nullptr;
  std::string op_type_;
  std::string op_domain_;
};


bool EnablePyCustomOps(bool enable = true);
