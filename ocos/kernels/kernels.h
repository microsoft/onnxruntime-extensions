// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

typedef OrtCustomOp const* CPTR_OrtCustomOp;
typedef CPTR_OrtCustomOp (*FxGetSchemaInstance)();

FxGetSchemaInstance const* GetCustomOpSchemaList();

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

#if defined(ENABLE_TOKENIZER)
const OrtCustomOp** LoadTokenizerSchemaList();
#endif  // ENABLE_TEXT_DOMAIN
