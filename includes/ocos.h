// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include <string>
#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>

#include "onnxruntime_customop.hpp"

// A helper API to support test kernels.
// Must be invoked before RegisterCustomOps.
extern "C" bool ORT_API_CALL AddExternalCustomOp(const OrtCustomOp* c_op);

constexpr const char* c_OpDomain = "ai.onnx.contrib";
constexpr const char* c_ComMsExtOpDomain = "com.microsoft.extensions";

struct BaseKernel {
  BaseKernel(const OrtApi& api, const OrtKernelInfo& info) noexcept : api_(api), info_(info), ort_(api_) {
  }

  template <class T>
  bool TryToGetAttribute(const char* name, T& value) const noexcept;

  template <class T>
  T TryToGetAttributeWithDefault(const char* name, const T& default_value) const noexcept {
    T result = default_value;
    TryToGetAttribute(name, result);
    return result;
  }

  void SetOutput(OrtKernelContext* ctx, size_t output_idx, const std::vector<int64_t>& dim,
                 const std::vector<int64_t>& data);

 protected:
  OrtErrorCode GetErrorCodeAndRelease(OrtStatusPtr status) const noexcept;
  const OrtApi& api_;
  OrtW::CustomOpApi ort_;
  const OrtKernelInfo& info_;
};

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions() = default;
  OrtTensorDimensions(const OrtW::CustomOpApi& ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }

  int64_t Size() const {
    int64_t s = 1;
    for (auto it = begin(); it != end(); ++it)
      s *= *it;
    return s;
  }

  bool IsScalar() const {
    return empty();
  }

  bool IsVector() const {
    return size() == 1;
  }
};

template <typename... Args>
class CuopContainer {
 public:
  CuopContainer() : op_instances_({[]() { return std::make_shared<Args>(); }()...}) {
    ocos_list_.reserve(op_instances_.size());
    std::transform(op_instances_.begin(), op_instances_.end(), std::back_inserter(ocos_list_),
                   [](const std::shared_ptr<OrtCustomOp>& custom_op) { return custom_op.get(); });
  }

  const std::vector<const OrtCustomOp*>& GetCustomOps() const {
    return ocos_list_;
  }

 private:
  std::vector<const OrtCustomOp*> ocos_list_;
  std::vector<std::shared_ptr<OrtCustomOp>> op_instances_;  // use shared_ptr to capture type specific deleter
};

#define CustomCpuFunc(name, f) []() { return std::shared_ptr<ortc::OrtLiteCustomOp>(ortc::CreateLiteCustomOp(name, "CPUExecutionProvider", f)); }
#define CustomCpuStruct(name, s) []() { return std::shared_ptr<ortc::OrtLiteCustomOp>(ortc::CreateLiteCustomOp<s>(name, "CPUExecutionProvider")); }
#define CustomAzureStruct(name, s) []() { return std::shared_ptr<ortc::OrtLiteCustomOp>(ortc::CreateLiteCustomOp<s>(name, "AzureExecutionProvider")); }

template <typename F>
void AppendCustomOp(std::vector<std::shared_ptr<OrtCustomOp>>& ops,
                    F arg) {
  ops.emplace_back(std::move(arg()));
}

template <typename T, typename... Args>
void AppendCustomOp(std::vector<std::shared_ptr<OrtCustomOp>>& ops,
                    T arg, Args... args) {
  AppendCustomOp(ops, arg);
  AppendCustomOp(ops, args...);
}

class OrtOpLoader {
 public:
  template <typename... Args>
  OrtOpLoader(Args... args) {
    LoadOps(args...);
    for (auto& ptr : op_instances_) {
      if (ptr)
        ocos_list_.push_back(ptr.get());
    }
  }

  const std::vector<const OrtCustomOp*>& GetCustomOps() const {
    return ocos_list_;
  }

 private:
  template <typename T>
  void LoadOps(T fn) {
    AppendCustomOp(op_instances_, fn);
  }

  template <typename T, typename... Args>
  void LoadOps(T fn, Args... args) {
    AppendCustomOp(op_instances_, fn);
    AppendCustomOp(op_instances_, args...);
  }

  std::vector<const OrtCustomOp*> ocos_list_;
  std::vector<std::shared_ptr<OrtCustomOp>> op_instances_;
};

struct CustomOpClassBegin {
};

using FxLoadCustomOpFactory = std::function<const std::vector<const OrtCustomOp*>&()>;

template <typename _Begin_place_holder, typename... Args>
const std::vector<const OrtCustomOp*>& LoadCustomOpClasses() {
  static CuopContainer<Args...> ctr;  // Let C++ runtime take cares of the MP initializing.
  return ctr.GetCustomOps();
}

#if defined(PYTHON_OP_SUPPORT)
const OrtCustomOp* FetchPyCustomOps(size_t& count);
OrtStatusPtr RegisterPythonDomainAndOps(OrtSessionOptions*, const OrtApi*);
#endif

#ifdef ENABLE_MATH
extern FxLoadCustomOpFactory LoadCustomOpClasses_Math;
#endif  // ENABLE_MATH

#ifdef ENABLE_TOKENIZER
extern FxLoadCustomOpFactory LoadCustomOpClasses_Tokenizer;
#endif  // ENABLE_TOKENIZER

#ifdef ENABLE_TF_STRING
extern FxLoadCustomOpFactory LoadCustomOpClasses_Text;
#endif  // ENABLE_TF_STRING

#ifdef ENABLE_CV2
extern FxLoadCustomOpFactory LoadCustomOpClasses_CV2;
#endif  // ENABLE_OPENCV

#ifdef ENABLE_VISION
extern FxLoadCustomOpFactory LoadCustomOpClasses_Vision;
#endif

#ifdef ENABLE_DR_LIBS
extern FxLoadCustomOpFactory LoadCustomOpClasses_Audio;
#endif

#if defined(ENABLE_AZURE) && ORT_API_VERSION >= 14
extern FxLoadCustomOpFactory LoadCustomOpClasses_Azure;
#endif
