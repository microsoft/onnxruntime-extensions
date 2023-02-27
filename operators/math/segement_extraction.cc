// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "segment_extraction.hpp"

KernelSegmentExtraction::KernelSegmentExtraction(const OrtApi& api, const OrtKernelInfo& info)
    : BaseKernel(api, info) {
}

void KernelSegmentExtraction::Compute(OrtKernelContext* context) {
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  const int64_t* p_data = ort_.GetTensorData<int64_t>(input);
  OrtTensorDimensions input_dim(ort_, input);
  if (!((input_dim.size() == 1) || (input_dim.size() == 2 && input_dim[0] == 1))) {
    ORTX_CXX_API_THROW("[SegmentExtraction]: Expect input dimension [n] or [1,n].", ORT_INVALID_GRAPH);
  }

  std::vector<std::int64_t> segment_value;
  std::vector<std::int64_t> segment_position;
  for (std::int64_t i = 0; i < input_dim.Size(); i++) {
    if (!p_data[i]) {
      continue;
    }

    // push start position and value
    if (i == 0 || p_data[i - 1] != p_data[i]) {
      segment_value.push_back(p_data[i]);
      segment_position.push_back(i);
    }

    // push end position
    if (i == (input_dim.Size() - 1) || p_data[i + 1] != p_data[i]) {
      segment_position.push_back(i + 1);
    }
  }

  std::vector<int64_t> segment_value_dim({static_cast<int64_t>(segment_value.size())});
  std::vector<int64_t> segment_position_dim({static_cast<int64_t>(segment_value.size()), 2});
  SetOutput(context, 0, segment_position_dim, segment_position);
  SetOutput(context, 1, segment_value_dim, segment_value);
}

size_t CustomOpSegmentExtraction::GetInputTypeCount() const {
  return 1;
};

size_t CustomOpSegmentExtraction::GetOutputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpSegmentExtraction::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

const char* CustomOpSegmentExtraction::GetName() const {
  return "SegmentExtraction";
};

ONNXTensorElementDataType CustomOpSegmentExtraction::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};
