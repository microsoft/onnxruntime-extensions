// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include "ocos.h"
#include "string_utils.h"

namespace std {

template <class T>
struct hash<std::vector<T>> {
  size_t operator()(const vector<T>& __vector) const noexcept;
};
}  // namespace std

class VectorToStringImpl {
 public:
  VectorToStringImpl(std::string& map, std::string& unk);
  std::vector<std::string> Compute(const void* input, const OrtTensorDimensions& input_dim, OrtTensorDimensions& output_dim);

 private:
  void ParseMappingTable(std::string& map);
  size_t ParseVectorLen(const std::string_view& line);
  void ParseValues(const std::string_view& v, std::vector<int64_t>& values);

  std::unordered_map<std::vector<int64_t>, std::string> map_;
  std::string unk_value_;
  size_t vector_len_;
};

struct KernelVectorToString : BaseKernel {
  KernelVectorToString(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context);

 private:
  std::shared_ptr<VectorToStringImpl> impl_;
};

struct CustomOpVectorToString : OrtW::CustomOpBase<CustomOpVectorToString, KernelVectorToString> {
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};
