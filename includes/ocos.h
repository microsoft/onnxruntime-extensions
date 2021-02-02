// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT


#if defined(ENABLE_GPT2_TOKENIZER)
const OrtCustomOp** LoadTokenizerSchemaList();
#endif  // ENABLE_GPT2_TOKENIZER


#if defined(PYTHON_OP_SUPPORT)
const OrtCustomOp* FetchPyCustomOps(size_t& count);
bool EnablePyCustomOps(bool enable=true);
#endif

// A helper API to support test kernels.
// Must be invoked before RegisterCustomOps.
extern "C" bool AddExternalCustomOp(const OrtCustomOp* c_op);

const char c_OpDomain[] = "ai.onnx.contrib";

struct BaseKernel {
  BaseKernel(OrtApi api) : api_(api), info_(nullptr), ort_(api_) {}
  BaseKernel(OrtApi api, const OrtKernelInfo *info) : api_(api), info_(info), ort_(api_) {}

 protected:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
  const OrtKernelInfo* info_;
};

struct OrtTensorDimensions : std::vector<int64_t> {
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
