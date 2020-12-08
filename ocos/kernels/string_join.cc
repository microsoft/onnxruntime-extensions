// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_join.hpp"

KernelStringJoin::KernelStringJoin(OrtApi api) : BaseKernel(api) {
}

void KernelStringJoin::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const std::string* X = ort_.GetTensorData<std::string>(input_X);
  const OrtValue* input_sep = ort_.KernelContext_GetInput(context, 1);
  const std::string* sep = ort_.GetTensorData<std::string>(input_sep);
  const OrtValue* input_axis = ort_.KernelContext_GetInput(context, 2);
  const int64_t* axis = ort_.GetTensorData<int64_t>(input_axis);

  // Setup output
  OrtTensorDimensions dimensions_sep(ort_, input_sep);
  if (dimensions_sep.size() != 1 || dimensions_sep[0] != 1)
    throw std::runtime_error("Input 2 is the separator, it has 1 element.");
  OrtTensorDimensions dimensions_axis(ort_, input_axis);
  if (dimensions_axis.size() != 1 || dimensions_axis[0] != 1)
    throw std::runtime_error("Input 3 is the axis, it has 1 element.");
  OrtTensorDimensions dimensions(ort_, input_X);
  if (*axis < 0 || *axis >= dimensions.size())
    throw std::runtime_error(MakeString("axis must be positive and smaller than the number of dimension but it is ", *axis));

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
  std::string* out = ort_.GetTensorMutableData<std::string>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

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
        st << X[index] << *sep;
      }
      st << X[index];
      out[pos] = st.str();
    }
  }
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
      throw std::runtime_error(MakeString("Unexpected input index ", index));
  }
};

size_t CustomOpStringJoin::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringJoin::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
