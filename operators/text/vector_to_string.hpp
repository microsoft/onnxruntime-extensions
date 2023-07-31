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
  std::vector<std::string> Compute(const void* input,
                                   const std::vector<int64_t>& input_dim,
                                   std::vector<int64_t>& output_dim) const;

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
  void Compute(const ortc::Tensor<int64_t>& input,
               ortc::Tensor<std::string>& out) const;

 private:
  std::shared_ptr<VectorToStringImpl> impl_;
};
