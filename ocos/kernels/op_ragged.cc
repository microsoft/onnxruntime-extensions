// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_ragged.hpp"
#include <string>

KernelRagged::KernelRagged(OrtApi api) : BaseKernel(api) {
}

void KernelRagged::Compute(OrtKernelContext* context) {
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

size_t CustomOpRagged::GetInputTypeCount() const {
  return 1;
};

size_t CustomOpRagged::GetOutputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpRagged::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

void* CustomOpRagged::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelRagged(api);
};

const char* CustomOpRagged::GetName() const {
  return "Ragged";
};

ONNXTensorElementDataType CustomOpRagged::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};
