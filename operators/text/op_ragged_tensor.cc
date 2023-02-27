// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "string_utils.h"
#include "string_tensor.h"
#include "op_ragged_tensor.hpp"

void KernelRaggedTensorToSparse::Compute(OrtKernelContext* context) {
  const OrtValue* n_elements = ort_.KernelContext_GetInput(context, 0);
  const int64_t* p_n_elements = ort_.GetTensorData<int64_t>(n_elements);

  OrtTensorDimensions d_length(ort_, n_elements);

  if (d_length.size() != 1)
    ORTX_CXX_API_THROW(MakeString(
                           "First input must have one dimension not ", d_length.size(), "."),
                       ORT_INVALID_ARGUMENT);
  int64_t n_els = d_length[0] - 1;
  int64_t n_values = p_n_elements[n_els];
  std::vector<int64_t> shape{n_values, 2};
  std::vector<int64_t> shape2{2};

  OrtValue* v0 = ort_.KernelContext_GetOutput(context, 0, shape.data(), shape.size());
  int64_t* out0 = ort_.GetTensorMutableData<int64_t>(v0);
  OrtValue* v1 = ort_.KernelContext_GetOutput(context, 1, shape2.data(), shape2.size());
  int64_t* out1 = ort_.GetTensorMutableData<int64_t>(v1);
  out1[0] = n_els;
  out1[1] = 0;
  int64_t row = 0;
  int64_t i, j, length;
  for (i = 1; i < d_length[0]; ++i) {
    length = p_n_elements[i] - p_n_elements[i - 1];
    if (length > out1[1])
      out1[1] = length;
    for (j = 0; j < length; ++j) {
      *out0++ = row;
      *out0++ = j;
    }
    ++row;
  }
}

size_t CustomOpRaggedTensorToSparse::GetInputTypeCount() const {
  return 1;
};

size_t CustomOpRaggedTensorToSparse::GetOutputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpRaggedTensorToSparse::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

const char* CustomOpRaggedTensorToSparse::GetName() const {
  return "RaggedTensorToSparse";
};

ONNXTensorElementDataType CustomOpRaggedTensorToSparse::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

CommonRaggedTensorToDense::CommonRaggedTensorToDense(const OrtApi& api, const OrtKernelInfo& info)
    : BaseKernel(api, info) {
}

void CommonRaggedTensorToDense::GetInputDims(OrtKernelContext* context, const OrtValue** inputs, OrtTensorDimensions* dims) {
  for (int i = 0; i < 4; ++i) {
    inputs[i] = ort_.KernelContext_GetInput(context, i);
    dims[i] = OrtTensorDimensions(ort_, inputs[i]);
  }
}

int64_t CommonRaggedTensorToDense::GetMaxCol(int64_t n, const int64_t* p_indices) {
  int64_t size = n;
  int64_t max_col = 0;
  for (int64_t i = 1; i < size; ++i) {
    max_col = std::max(max_col, p_indices[i] - p_indices[i - 1]);
  }
  return max_col;
}

KernelRaggedTensorToDense::KernelRaggedTensorToDense(const OrtApi& api, const OrtKernelInfo& info)
    : CommonRaggedTensorToDense(api, info) {
  missing_value_ = HasAttribute("missing_value") ? ort_.KernelInfoGetAttribute<int64_t>(&info, "missing_value") : -1;
}

void KernelRaggedTensorToDense::Compute(OrtKernelContext* context) {
  const OrtValue* inputs[4];
  OrtTensorDimensions dims[4];
  GetInputDims(context, inputs, dims);

  const int64_t* p_values = ort_.GetTensorData<int64_t>(inputs[1]);
  const int64_t* p_missing = ort_.GetTensorData<int64_t>(inputs[2]);
  const int64_t* p_indices = ort_.GetTensorData<int64_t>(inputs[3]);

  int64_t size = dims[3].Size();
  int64_t max_col = GetMaxCol(size, p_indices);

  std::vector<int64_t> shape_out{size - 1, max_col};
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, shape_out.data(), shape_out.size());
  int64_t* dense = ort_.GetTensorMutableData<int64_t>(output);

  int64_t pos = 0;
  int64_t j, pos_end;
  int64_t shape_out_size = shape_out[0] * shape_out[1];
  for (int64_t i = 0; i < size - 1; ++i) {
    pos_end = pos + max_col;
    if (pos_end > shape_out_size)
      ORTX_CXX_API_THROW(MakeString(
                             "Unexpected index ", pos_end, " greather than ", shape_out[0], "x", shape_out[1],
                             " - i=", i, " size=", size, "."),
                         ORT_INVALID_ARGUMENT);
    for (j = p_indices[i]; j < p_indices[i + 1]; ++j, ++pos) {
      dense[pos] = p_values[j];
    }
    for (; pos < pos_end; ++pos) {
      dense[pos] = p_missing[0];
    }
  }
}

size_t CustomOpRaggedTensorToDense::GetInputTypeCount() const {
  return 4;
};

size_t CustomOpRaggedTensorToDense::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpRaggedTensorToDense::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

const char* CustomOpRaggedTensorToDense::GetName() const {
  return "RaggedTensorToDense";
};

ONNXTensorElementDataType CustomOpRaggedTensorToDense::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

KernelStringRaggedTensorToDense::KernelStringRaggedTensorToDense(const OrtApi& api, const OrtKernelInfo& info) : CommonRaggedTensorToDense(api, info) {
}

void KernelStringRaggedTensorToDense::Compute(OrtKernelContext* context) {
  const OrtValue* inputs[4];
  OrtTensorDimensions dims[4];
  GetInputDims(context, inputs, dims);

  std::vector<std::string> input;
  GetTensorMutableDataString(api_, ort_, context, inputs[1], input);
  const int64_t* p_indices = ort_.GetTensorData<int64_t>(inputs[3]);
  int64_t size = dims[3].Size();
  int64_t max_col = GetMaxCol(size, p_indices);
  std::vector<int64_t> shape_out{size - 1, max_col};

  int64_t shape_out_size = shape_out[0] * shape_out[1];
  std::vector<std::string> dense(static_cast<size_t>(max_col * (size - 1)));
  int64_t pos = 0;
  int64_t j, pos_end;
  for (int64_t i = 0; i < size - 1; ++i) {
    pos_end = pos + max_col;
    if (pos_end > shape_out_size)
      ORTX_CXX_API_THROW(MakeString(
                             "Unexpected index ", pos_end, " greather than ", shape_out[0], "x", shape_out[1],
                             " - i=", i, " size=", size, "."),
                         ORT_INVALID_ARGUMENT);
    for (j = p_indices[i]; j < p_indices[i + 1]; ++j, ++pos) {
      dense[static_cast<size_t>(pos)] = input[static_cast<size_t>(j)];
    }
    pos = pos_end;
  }

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, shape_out.data(), shape_out.size());
  FillTensorDataString(api_, ort_, context, dense, output);
}

size_t CustomOpStringRaggedTensorToDense::GetInputTypeCount() const {
  return 4;
};

size_t CustomOpStringRaggedTensorToDense::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringRaggedTensorToDense::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

const char* CustomOpStringRaggedTensorToDense::GetName() const {
  return "StringRaggedTensorToDense";
};

ONNXTensorElementDataType CustomOpStringRaggedTensorToDense::GetInputType(size_t index) const {
  switch (index) {
    case 1:
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 0:
    case 3:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      ORTX_CXX_API_THROW(MakeString("[StringRaggedTensorToDense] Unexpected output index ", index, "."), ORT_INVALID_ARGUMENT);
  }
};
