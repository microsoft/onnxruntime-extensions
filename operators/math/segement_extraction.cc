// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "segment_extraction.hpp"

OrtStatusPtr segment_extraction(const ortc::Tensor<int64_t>& input,
                        ortc::Tensor<int64_t>& output0,
                        ortc::Tensor<int64_t>& output1) {
  auto& input_dim = input.Shape();
  if (!((input_dim.size() == 1) || (input_dim.size() == 2 && input_dim[0] == 1))) {
    return OrtW::CreateStatus("[SegmentExtraction]: Expect input dimension [n] or [1,n].", ORT_INVALID_GRAPH);
  }
  const int64_t* p_data = input.Data();
  std::vector<std::int64_t> segment_value;
  std::vector<std::int64_t> segment_position;
  for (std::int64_t i = 0; i < input.NumberOfElement(); i++) {
    if (!p_data[i]) {
      continue;
    }

    // push start position and value
    if (i == 0 || p_data[i - 1] != p_data[i]) {
      segment_value.push_back(p_data[i]);
      segment_position.push_back(i);
    }

    // push end position
    if (i == (input.NumberOfElement() - 1) || p_data[i + 1] != p_data[i]) {
      segment_position.push_back(i + 1);
    }
  }

  std::vector<int64_t> segment_value_dim({static_cast<int64_t>(segment_value.size())});
  std::vector<int64_t> segment_position_dim({static_cast<int64_t>(segment_value.size()), 2});

  int64_t* out0_data = output0.Allocate(segment_position_dim);
  std::copy(segment_position.begin(), segment_position.end(), out0_data);

  int64_t* out1_data = output1.Allocate(segment_value_dim);
  std::copy(segment_value.begin(), segment_value.end(), out1_data);
  return nullptr;
}
