// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "segment_sum.hpp"

template <typename T>
void KernelSegmentSum_Compute(OrtW::CustomOpApi& ort_, OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* data = ort_.KernelContext_GetInput(context, 0);
  const T* p_data = ort_.GetTensorData<T>(data);
  const OrtValue* segment_ids = ort_.KernelContext_GetInput(context, 1);
  const int64_t* p_segment_ids = ort_.GetTensorData<int64_t>(segment_ids);

  // Setup output
  OrtTensorDimensions dim_data(ort_, data);
  OrtTensorDimensions dim_seg(ort_, segment_ids);
  if (dim_data.size() == 0 || dim_seg.size() == 0)
    ORTX_CXX_API_THROW("Both inputs cannot be empty.", ORT_INVALID_GRAPH);
  if (dim_seg.size() != 1)
    ORTX_CXX_API_THROW("segment_ids must a single tensor", ORT_INVALID_GRAPH);
  if (dim_data[0] != dim_seg[0])
    ORTX_CXX_API_THROW(MakeString(
                           "First dimensions of data and segment_ids should be the same, data shape: ", dim_data,
                           " segment_ids shape: ", dim_seg),
                       ORT_INVALID_GRAPH);

  int64_t last_seg = p_segment_ids[dim_seg[0] - 1];
  OrtTensorDimensions dim_out = dim_data;
  dim_out[0] = last_seg + 1;

  OrtValue* v = ort_.KernelContext_GetOutput(context, 0, dim_out.data(), dim_out.size());
  T* p_output = ort_.GetTensorMutableData<T>(v);
  int64_t out_size = dim_out.Size();
  memset(p_output, 0, static_cast<size_t>(out_size * sizeof(T)));

  // The implementation is naive. It could be parallelized and
  // use SIMD instructions to be faster.
  int64_t in_stride = dim_data.Size();
  const T* begin = p_data;
  const T* end = p_data + in_stride;
  in_stride /= dim_data[0];
  T *p_out, *p_out_end;
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

KernelSegmentSum::KernelSegmentSum(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
}

void KernelSegmentSum::Compute(OrtKernelContext* context) {
  KernelSegmentSum_Compute<float>(ort_, context);
}

size_t CustomOpSegmentSum::GetInputTypeCount() const {
  return 2;
};

size_t CustomOpSegmentSum::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpSegmentSum::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

const char* CustomOpSegmentSum::GetName() const {
  return "SegmentSum";
};

ONNXTensorElementDataType CustomOpSegmentSum::GetInputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      ORTX_CXX_API_THROW("Operator SegmentSum has 2 inputs.", ORT_INVALID_ARGUMENT);
  }
};
