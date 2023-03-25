// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "segment_sum.hpp"

void segment_sum(const ortc::TensorT<float>& data,
                 const ortc::TensorT<int64_t>& segment_ids,
                 ortc::TensorT<float>& output) {
  auto& dim_data = data.Shape();
  auto& dim_seg = segment_ids.Shape();
  if (dim_data.size() == 0 || dim_seg.size() == 0)
    ORTX_CXX_API_THROW("Both inputs cannot be empty.", ORT_INVALID_GRAPH);
  if (dim_seg.size() != 1)
    ORTX_CXX_API_THROW("segment_ids must a single tensor", ORT_INVALID_GRAPH);
  if (dim_data[0] != dim_seg[0])
    ORTX_CXX_API_THROW(MakeString(
                           "First dimensions of data and segment_ids should be the same, data shape: ", dim_data,
                           " segment_ids shape: ", dim_seg),
                       ORT_INVALID_GRAPH);

  const int64_t* p_segment_ids = segment_ids.Data();
  const float* p_data = data.Data();

  int64_t last_seg = p_segment_ids[dim_seg[0] - 1];
  std::vector<int64_t> dim_out = dim_data;
  dim_out[0] = last_seg + 1;

  float* p_output = output.Allocate(dim_out);
  int64_t out_size = output.NumerOfElement();
  memset(p_output, 0, static_cast<size_t>(out_size * sizeof(float)));

  // The implementation is naive. It could be parallelized and
  // use SIMD instructions to be faster.
  int64_t in_stride = data.NumerOfElement();
  const float* begin = p_data;
  const float* end = p_data + in_stride;
  in_stride /= dim_data[0];
  float *p_out, *p_out_end;
  const int64_t* p_seg = p_segment_ids;
  for (; begin != end; ++p_seg) {
    if ((p_seg != p_segment_ids) && (*p_seg != *(p_seg - 1)) && (*p_seg != *(p_seg - 1) + 1))
      ORTX_CXX_API_THROW(MakeString("segment_ids must be increasing but found ",
                                    *(p_seg - 1), " and ", *p_seg, " at position ",
                                    std::distance(p_segment_ids, p_seg), "."),
                         ORT_RUNTIME_EXCEPTION);
    p_out = p_output + *p_seg * in_stride;
    p_out_end = p_out + in_stride;
    for (; p_out != p_out_end; ++p_out, ++begin)
      *p_out += *begin;
  }
}
