// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

typedef const OrtCustomOp** (*FxLoadCustomOpFactory)();

#if defined(ENABLE_GPT2_TOKENIZER)
const OrtCustomOp** LoadTokenizerSchemaList();
#endif  // ENABLE_GPT2_TOKENIZER

#if defined(PYTHON_OP_SUPPORT)
const OrtCustomOp* FetchPyCustomOps(size_t& count);
bool EnablePyCustomOps(bool enable = true);
#endif


// A helper API to support test kernels.
// Must be invoked before RegisterCustomOps.
extern "C" bool AddExternalCustomOp(const OrtCustomOp* c_op);

const char c_OpDomain[] = "ai.onnx.contrib";

struct BaseKernel {
  BaseKernel(OrtApi api) : api_(api), info_(nullptr), ort_(api_) {}
  BaseKernel(OrtApi api, const OrtKernelInfo* info) : api_(api), info_(info), ort_(api_) {}

  bool HasAttribute(const char* name) const;
  template <class T>
  bool TryToGetAttribute(const char* name, T& value);
  template <class T>
  T TryToGetAttributeWithDefault(const char* name, T default_value);
 protected:
  OrtErrorCode GetErrorCodeAndRelease(OrtStatusPtr status);
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
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
  const std::vector<int64_t>& GetDims() const { return *this; }
  int64_t Size() const {
    int64_t s = 1.;
    for (auto it = begin(); it != end(); ++it)
      s *= *it;
    return s;
  }
};

template <class... Args>
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

template <typename... Args>
const OrtCustomOp** LoadCustomOpClasses() {
  static CuopContainer<Args...> ctr;  // Let C++ runtime take cares of the MP initializing.
  return ctr.GetList();
}
