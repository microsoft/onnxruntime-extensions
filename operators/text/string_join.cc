// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_join.hpp"
#include "string_tensor.h"

KernelStringJoin::KernelStringJoin(OrtApi api) : BaseKernel(api) {
}

void KernelStringJoin::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_sep = ort_.KernelContext_GetInput(context, 1);
  const OrtValue* input_axis = ort_.KernelContext_GetInput(context, 2);
  const int64_t* axis = ort_.GetTensorData<int64_t>(input_axis);
  std::vector<std::string> X, sep;
  GetTensorMutableDataString(api_, ort_, context, input_X, X);
  GetTensorMutableDataString(api_, ort_, context, input_sep, sep);

  // Setup output
  OrtTensorDimensions dimensions_sep(ort_, input_sep);
  if (dimensions_sep.size() != 1 || dimensions_sep[0] != 1)
    ORT_CXX_API_THROW("Input 2 is the separator, it has 1 element.", ORT_INVALID_ARGUMENT);
  OrtTensorDimensions dimensions_axis(ort_, input_axis);
  if (dimensions_axis.size() != 1 || dimensions_axis[0] != 1)
    ORT_CXX_API_THROW("Input 3 is the axis, it has 1 element.", ORT_INVALID_ARGUMENT);
  OrtTensorDimensions dimensions(ort_, input_X);
  if (*axis < 0 || *axis >= dimensions.size())
    ORT_CXX_API_THROW(MakeString("axis must be positive and smaller than the number of dimension but it is ", *axis), ORT_INVALID_ARGUMENT);

  std::vector<int64_t> dimensions_out(dimensions.size() > 1 ? dimensions.size() - 1 : 1);
  if (dimensions.size() > 1) {
    for (size_t i = 0, pos = 0; i < dimensions.size(); ++i) {
      if (static_cast<int64_t>(i) == *axis)
        continue;
      dimensions_out[pos++] = dimensions[i];
    }
  } else {
    dimensions_out[0] = 1;
  }

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions_out.data(), dimensions_out.size());
  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);
  std::vector<std::string> out(size);

  // Do computation
  int64_t h = 1;
  for (size_t i = *axis + 1; i < dimensions.size(); ++i) {
    h *= dimensions[i];
  }
  int64_t left_part = size / h;
  int64_t right_part = size / left_part;
  int64_t n_red = dimensions[*axis] - 1;
  int64_t inc = right_part * (n_red + 1);
  int64_t pos = 0;
  for (int64_t li = 0; li < left_part; ++li) {
    for (int64_t ri = 0; ri < right_part; ++ri, ++pos) {
      std::ostringstream st;
      int64_t index = ri + li * inc;
      for (int64_t j = 0; j < n_red; ++j, index += h) {
        st << X[index] << sep[0];
      }
      st << X[index];
      out[pos] = st.str();
    }
  }
  FillTensorDataString(api_, ort_, context, out, output);
}

void* CustomOpStringJoin::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelStringJoin(api);
};

const char* CustomOpStringJoin::GetName() const {
  return "StringJoin";
};

size_t CustomOpStringJoin::GetInputTypeCount() const {
  return 3;
};

ONNXTensorElementDataType CustomOpStringJoin::GetInputType(size_t index) const {
  switch (index) {
    case 0:
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      ORT_CXX_API_THROW(MakeString("Unexpected input index ", index), ORT_INVALID_ARGUMENT);
  }
};

size_t CustomOpStringJoin::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringJoin::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
