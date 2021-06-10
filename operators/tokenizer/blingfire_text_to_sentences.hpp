// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelTextToSentences : BaseKernel {
  KernelTextToSentences(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);\
private:
 std::shared_ptr<void> model_;
};

struct CustomOpTextToSentences : Ort::CustomOpBase<CustomOpTextToSentences, KernelTextToSentences> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

extern "C" const int TextToSentencesWithOffsetsWithModel(
    const char* pInUtf8Str, int InUtf8StrByteCount,
    char* pOutUtf8Str, int* pStartOffsets, int* pEndOffsets,
    const int MaxOutUtf8StrByteCount, void* hModel);

extern "C"
int FreeModel(void* ModelPtr);

extern "C"
void* SetModel(const unsigned char * pImgBytes, int ModelByteCount);