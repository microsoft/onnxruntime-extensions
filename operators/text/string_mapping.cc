// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_mapping.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>

OrtStatusPtr KernelStringMapping::OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
  std::string map;
  auto status = OrtW::GetOpAttribute(info, "map", map);
  if (status != nullptr) {
    return status;
  }

  auto lines = SplitString(map, "\n", true);
  for (const auto& line : lines) {
    auto items = SplitString(line, "\t", true);

    if (items.size() != 2) {
      return OrtW::CreateStatus(
          ("[StringMapping]: Should only exist two items in one line, find error in line: " + std::string(line)).c_str(),
          ORT_INVALID_GRAPH);
    }
    map_[std::string(items[0])] = std::string(items[1]);
  }

  return nullptr;
}

OrtStatusPtr KernelStringMapping::Compute(const ortc::Tensor<std::string>& input,
                                          ortc::Tensor<std::string>& output) const {
  // make a copy as input is constant
  std::vector<std::string> input_data = input.Data();

  for (auto& str : input_data) {
    auto entry = map_.find(str);
    if (entry != map_.end()) {
      str = entry->second;
    }
  }
  output.SetStringOutput(input_data, input.Shape());
  return nullptr;
}
