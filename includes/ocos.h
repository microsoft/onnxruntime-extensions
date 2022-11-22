// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

// A helper API to support test kernels.
// Must be invoked before RegisterCustomOps.
extern "C" bool ORT_API_CALL AddExternalCustomOp(const OrtCustomOp* c_op);

constexpr const char* c_OpDomain = "ai.onnx.contrib";
constexpr const char* c_ComMsExtOpDomain = "com.microsoft.extensions";

struct BaseKernel {
  BaseKernel(const OrtApi& api) : api_(api), info_(nullptr), ort_(api_) {}
  BaseKernel(const OrtApi& api, const OrtKernelInfo* info) : api_(api), info_(info), ort_(api_) {}

  bool HasAttribute(const char* name) const;

  template <class T>
  bool TryToGetAttribute(const char* name, T& value);

  template <class T>
  T TryToGetAttributeWithDefault(const char* name, T default_value) {
    T& result = default_value;
    TryToGetAttribute(name, result);
    return result;
  }

  void SetOutput(OrtKernelContext* ctx, size_t output_idx, const std::vector<int64_t>& dim, const std::vector<int64_t>& data);

 protected:
  OrtErrorCode GetErrorCodeAndRelease(OrtStatusPtr status);
  const OrtApi& api_;
  Ort::CustomOpApi ort_;
  const OrtKernelInfo* info_;
};

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions() = default;
  OrtTensorDimensions(Ort::CustomOpApi& ort, const OrtValue* value) {
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
