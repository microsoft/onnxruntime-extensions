// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <vector>
#include "ocos.h"
#include "ustring.h"
#include "string_utils.h"
#include "string_tensor.h"


class BertTokenizerDecoder {
 public:
  BertTokenizerDecoder(std::string vocab, bool do_lower_case, bool do_basic_tokenize,
                       ustring unk_token, ustring sep_token, ustring pad_token, ustring  cls_token,
                       ustring mask_token, bool tokenize_chinese_chars, bool strip_accents,
                       ustring suffix_indicator);
  std::vector<int64_t> Decode(const std::vector<ustring>& tokens);

 private:
  int32_t unk_token_id_;
  int32_t sep_token_id_;
  int32_t pad_token_id_;
  int32_t cls_token_id_;
  int32_t mask_token_id_;
  std::shared_ptr<std::unordered_map<int32_t, ustring>> vocab_;
};

struct KernelBertTokenizerDecoder : BaseKernel {
  KernelBertTokenizerDecoder(OrtApi api,  const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);
 private:
  std::shared_ptr<BertTokenizerDecoder> decoder_;
};

struct CustomOpBertTokenizerDecoder : Ort::CustomOpBase<CustomOpBertTokenizerDecoder, KernelBertTokenizerDecoder> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};