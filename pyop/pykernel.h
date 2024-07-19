// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

#include <vector>
#include <map>
#include <memory>

struct PyCustomOpDef {
  std::string op_type;
  uint64_t obj_id = 0;
  std::vector<int> input_types;
  std::vector<int> output_types;
  std::map<std::string, int> attrs;

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
  PyCustomOpKernel(const OrtApi& api, const OrtKernelInfo& info, uint64_t id, const std::map<std::string, int>& attrs);
  void Compute(OrtKernelContext* context);

 private:
  const OrtApi& api_;
  OrtW::CustomOpApi ort_;
  uint64_t obj_id_;
  std::map<std::string, std::string> attrs_values_;
};

struct PyCustomOpFactory : public OrtCustomOp {

  PyCustomOpFactory() = default;

  PyCustomOpFactory(const PyCustomOpDef* opdef, const std::string& domain, const std::string& op) {
    if (opdef == nullptr)
      throw std::runtime_error("Python definition is empty.");
    opdef_ = opdef;
    op_domain_ = domain;
    op_type_ = op;

    OrtCustomOp::version = MIN_ORT_VERSION_SUPPORTED;  // The minimum ORT version supported
    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi* api, const OrtKernelInfo* info) {
      void* p = nullptr;

      OCOS_API_IMPL_BEGIN
      auto self = static_cast<const PyCustomOpFactory*>(this_);
      auto kernel = std::make_unique<PyCustomOpKernel>(*api, *info, self->opdef_->obj_id, self->opdef_->attrs).release();
      p = reinterpret_cast<void*>(kernel);
      OCOS_API_IMPL_END

      return p;
    };

    OrtCustomOp::GetName = [](const OrtCustomOp* this_) noexcept {
      auto self = static_cast<const PyCustomOpFactory*>(this_);
      return self->op_type_.c_str();
    };

    OrtCustomOp::GetExecutionProviderType = [](const OrtCustomOp* this_) noexcept {
      return "CPUExecutionProvider";
    };

    OrtCustomOp::GetInputTypeCount = [](const OrtCustomOp* this_) noexcept {
      auto self = static_cast<const PyCustomOpFactory*>(this_);
      return self->opdef_->input_types.size();
    };

    OrtCustomOp::GetInputType = [](const OrtCustomOp* this_, size_t index) noexcept {
      auto self = static_cast<const PyCustomOpFactory*>(this_);
      return static_cast<ONNXTensorElementDataType>(self->opdef_->input_types[index]);
    };

    OrtCustomOp::GetOutputTypeCount = [](const OrtCustomOp* this_) noexcept {
      auto self = static_cast<const PyCustomOpFactory*>(this_);
      return self->opdef_->output_types.size();
    };

    OrtCustomOp::GetOutputType = [](const OrtCustomOp* this_, size_t index) noexcept {
      auto self = static_cast<const PyCustomOpFactory*>(this_);
      return static_cast<ONNXTensorElementDataType>(self->opdef_->output_types[index]);
    };

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) noexcept {
      OCOS_API_IMPL_BEGIN
      static_cast<PyCustomOpKernel*>(op_kernel)->Compute(context);
      OCOS_API_IMPL_END
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) noexcept {
      std::unique_ptr<PyCustomOpKernel>(reinterpret_cast<PyCustomOpKernel*>(op_kernel)).reset();
    };

    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp* this_, size_t index) noexcept {
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    };

    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp* this_, size_t index) noexcept {
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    };
  }

  const PyCustomOpDef* opdef_ = nullptr;
  std::string op_type_;
  std::string op_domain_;
};

bool EnablePyCustomOps(bool enable = true);

void AddGlobalMethodsCApi(pybind11::module& m);
