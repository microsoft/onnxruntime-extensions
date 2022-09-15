// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <functional>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT


// A helper API to support test kernels.
// Must be invoked before RegisterCustomOps.
extern "C" bool ORT_API_CALL AddExternalCustomOp(const OrtCustomOp* c_op);

const char c_OpDomain[] = "ai.onnx.contrib";

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

  void SetOutput(OrtKernelContext* ctx,  size_t output_idx, const std::vector<int64_t>& dim, const std::vector<int64_t>& data);

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

  bool IsScalar() const{
    return empty();
  }

  bool IsVector() const{
    return size() == 1;
  }
};


template <typename... Args>
class CuopContainer {
 public:
  CuopContainer() : ocos_list_({[]() { return new Args; }()...}) {
    ocos_list_.push_back(nullptr);
  }

  ~CuopContainer() {
    // skip the last null pointer.
    for (auto i = 0; i < ocos_list_.size() - 1; i++) {
      delete ocos_list_[i];
    }

    ocos_list_.clear();
  }

  const OrtCustomOp** GetList() {
    return &const_cast<const OrtCustomOp*&>(ocos_list_.front());
  }

 private:
  std::vector<OrtCustomOp*> ocos_list_;
};

struct CustomOpClassBegin{
};

typedef std::function<const OrtCustomOp**()> FxLoadCustomOpFactory;

template <typename _Begin_place_holder, typename... Args>
const OrtCustomOp** LoadCustomOpClasses() {
  static CuopContainer<Args...> ctr;  // Let C++ runtime take cares of the MP initializing.
  return ctr.GetList();
}

#if defined(PYTHON_OP_SUPPORT)
const OrtCustomOp* FetchPyCustomOps(size_t& count);
OrtStatusPtr RegisterPythonDomainAndOps(OrtSessionOptions*, const OrtApi*);
bool EnablePyCustomOps(bool enable = true);
#endif

#ifdef ENABLE_MATH
extern FxLoadCustomOpFactory LoadCustomOpClasses_Math;
#endif  // ENABLE_MATH

#ifdef ENABLE_TOKENIZER
extern FxLoadCustomOpFactory LoadCustomOpClasses_Tokenizer;
#endif // ENABLE_TOKENIZER

#ifdef ENABLE_TF_STRING
extern FxLoadCustomOpFactory LoadCustomOpClasses_Text;
#endif  // ENABLE_TF_STRING

#ifdef ENABLE_OPENCV
extern FxLoadCustomOpFactory LoadCustomOpClasses_OpenCV;
#endif  // ENABLE_OPENCV
