// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include "ocos.h"
#include "string_utils.h"

class StringToVectorImpl {
 public:
  StringToVectorImpl(std::string& map, std::string& unk);
  std::vector<std::vector<int64_t>> Compute(std::vector<std::string>& str_input, const OrtTensorDimensions& input_dim, OrtTensorDimensions& output_dim);

 private:
  void ParseMappingTable(std::string& map);
  void ParseUnkownValue(std::string& unk);
  size_t ParseVectorLen(const std::string_view& line);
  void ParseValues(const std::string_view& v, std::vector<int64_t>& values);

  // mapping of string to vector
  std::unordered_map<std::string, std::vector<int64_t>> map_;
  // unkown value is a vector of int
  std::vector<int64_t> unk_value_;
  size_t vector_len_;
};

struct KernelStringToVector : BaseKernel {
  KernelStringToVector(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context);

 private:
  std::shared_ptr<StringToVectorImpl> impl_;
};

struct CustomOpStringToVector : OrtW::CustomOpBase<CustomOpStringToVector, KernelStringToVector> {
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};
