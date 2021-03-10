// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_ragged_tensor.hpp"
#include "string_common.h"

KernelRaggedTensorToSparse::KernelRaggedTensorToSparse(OrtApi api) : BaseKernel(api) {
}

void KernelRaggedTensorToSparse::Compute(OrtKernelContext* context) {
  const OrtValue* n_elements = ort_.KernelContext_GetInput(context, 0);
  const int64_t* p_n_elements = ort_.GetTensorData<int64_t>(n_elements);

  OrtTensorDimensions d_length(ort_, n_elements);

  if (d_length.size() != 1)
    throw std::runtime_error(MakeString(
        "First input must have one dimension not ", d_length.size(), "."));
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

void* CustomOpRaggedTensorToSparse::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelRaggedTensorToSparse(api);
};

const char* CustomOpRaggedTensorToSparse::GetName() const {
  return "RaggedTensorToSparse";
};

ONNXTensorElementDataType CustomOpRaggedTensorToSparse::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

KernelRaggedTensorToDense::KernelRaggedTensorToDense(OrtApi api) : BaseKernel(api) {
}

void KernelRaggedTensorToDense::Compute(OrtKernelContext* context) {
  const OrtValue* inputs[5];
  OrtTensorDimensions dims[5];
  for (int i = 0; i < 5; ++i) {
    inputs[i] = ort_.KernelContext_GetInput(context, i);
    dims[i] = OrtTensorDimensions(ort_, inputs[i]);
  }

  const int64_t* p_in0 = ort_.GetTensorData<int64_t>(inputs[0]);
  const int64_t* p_in3 = ort_.GetTensorData<int64_t>(inputs[3]);
  const int64_t* p_in4 = ort_.GetTensorData<int64_t>(inputs[4]);

  std::vector<std::string> in1, in2;
  GetTensorMutableDataString(api_, ort_, context, inputs[1], in1);
  GetTensorMutableDataString(api_, ort_, context, inputs[2], in2);

  int64_t size = dims[4].Size();
  int64_t max_col = 0;
  for (int64_t i = 1; i < size; ++i) {
    max_col = std::max(max_col, p_in4[i] - p_in4[i - 1]);
  }
  std::vector<std::string> dense(max_col * (size - 1));
  int64_t pos = 0;
  int64_t j, pos_end;
  for (int64_t i = 0; i < size - 1; ++i) {
    pos_end = pos + max_col;
    for (j = p_in4[i]; j < p_in4[i + 1]; ++j, ++pos) {
      dense[pos] = in1[j];
    }
    pos = pos_end;
  }

  std::vector<int64_t> shape_out{size - 1, max_col};
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, shape_out.data(), shape_out.size());
  FillTensorDataString(api_, ort_, context, dense, output);
}

size_t CustomOpRaggedTensorToDense::GetInputTypeCount() const {
  return 5;
};

size_t CustomOpRaggedTensorToDense::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpRaggedTensorToDense::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

void* CustomOpRaggedTensorToDense::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelRaggedTensorToDense(api);
};

const char* CustomOpRaggedTensorToDense::GetName() const {
  return "RaggedTensorToDense";
};

ONNXTensorElementDataType CustomOpRaggedTensorToDense::GetInputType(size_t index) const {
  switch (index) {
    case 0:
    case 3:
    case 4:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case 1:
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    default:
      throw std::runtime_error(MakeString("Unexpected output index ", index, "."));
  }
};
