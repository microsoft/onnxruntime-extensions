// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_mapping.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>

KernelStringMapping::KernelStringMapping(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  std::string map = ort_.KernelInfoGetAttribute<std::string>(&info, "map");
  auto lines = SplitString(map, "\n", true);
  for (const auto& line : lines) {
    auto items = SplitString(line, "\t", true);

    if (items.size() != 2) {
      ORTX_CXX_API_THROW(std::string("[StringMapping]: Should only exist two items in one line, find error in line: ") + std::string(line), ORT_INVALID_GRAPH);
    }
    map_[std::string(items[0])] = std::string(items[1]);
  }
}

void KernelStringMapping::Compute(const ortc::Tensor<std::string>& input,
                                  ortc::Tensor<std::string>& output) {
  // make a copy as input is constant
  std::vector<std::string> input_data = input.Data();

  for (auto& str : input_data) {
    if (map_.find(str) != map_.end()) {
      str = map_[str];
    }
  }
  output.SetStringOutput(0, input_data, input.Shape());
}
