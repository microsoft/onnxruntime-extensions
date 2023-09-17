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
  std::vector<std::vector<int64_t>> Compute(const std::vector<std::string>& str_input,
                                            const std::vector<int64_t>& input_dim,
                                            std::vector<int64_t>& output_dim);

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
  void Compute(const ortc::Tensor<std::string>& input,
               ortc::Tensor<int64_t>& out);

 private:
  std::shared_ptr<StringToVectorImpl> impl_;
};
