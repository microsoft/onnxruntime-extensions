// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "string_utils.h"
#include "string_tensor.h"
#include "op_ragged_tensor.hpp"

void KernelRaggedTensoroSparse::Compute(const ortc::Tensor<int64_t>& n_element,
                                         ortc::Tensor<int64_t>& output_0,
                                         ortc::Tensor<int64_t>& output_1) {
  const int64_t* p_n_elements = n_element.Data();

  auto& d_length = n_element.Shape();

  if (d_length.size() != 1)
    ORTX_CXX_API_THROW(MakeString(
                           "First input must have one dimension not ", d_length.size(), "."),
                       ORT_INVALID_ARGUMENT);
  int64_t n_els = d_length[0] - 1;
  int64_t n_values = p_n_elements[n_els];
  std::vector<int64_t> shape{n_values, 2};
  std::vector<int64_t> shape2{2};

  int64_t* out0 = output_0.Allocate(shape);
  int64_t* out1 = output_1.Allocate(shape2);
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

CommonRaggedTensoroDense::CommonRaggedTensoroDense(const OrtApi& api, const OrtKernelInfo& info)
    : BaseKernel(api, info) {
}

void CommonRaggedTensoroDense::GetInputDims(OrtKernelContext* context, const OrtValue** inputs, OrtTensorDimensions* dims) {
  for (int i = 0; i < 4; ++i) {
    inputs[i] = ort_.KernelContext_GetInput(context, i);
    dims[i] = OrtTensorDimensions(ort_, inputs[i]);
  }
}

int64_t CommonRaggedTensoroDense::GetMaxCol(int64_t n, const int64_t* p_indices) {
  int64_t size = n;
  int64_t max_col = 0;
  for (int64_t i = 1; i < size; ++i) {
    max_col = std::max(max_col, p_indices[i] - p_indices[i - 1]);
  }
  return max_col;
}

KernelRaggedTensoroDense::KernelRaggedTensoroDense(const OrtApi& api, const OrtKernelInfo& info)
    : CommonRaggedTensoroDense(api, info) {
  missing_value_ = TryToGetAttributeWithDefault("missing_value", -1);
}

void KernelRaggedTensoroDense::Compute(const ortc::Tensor<int64_t>& input0,
                                        const ortc::Tensor<int64_t>& input1,
                                        const ortc::Tensor<int64_t>& input2,
                                        const ortc::Tensor<int64_t>& input3,
                                        ortc::Tensor<int64_t>& output) {
  const int64_t* p_values = input1.Data();
  const int64_t* p_missing = input2.Data();
  const int64_t* p_indices = input3.Data();

  int64_t size = input3.NumberOfElement();
  int64_t max_col = GetMaxCol(size, p_indices);

  std::vector<int64_t> shape_out{size - 1, max_col};
  int64_t* dense = output.Allocate(shape_out);

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

KernelStringRaggedTensoroDense::KernelStringRaggedTensoroDense(const OrtApi& api, const OrtKernelInfo& info) : CommonRaggedTensoroDense(api, info) {
}

void KernelStringRaggedTensoroDense::Compute(const ortc::Tensor<int64_t>& input0,
                                              const ortc::Tensor<std::string>& input1,
                                              const ortc::Tensor<int64_t>& input2,
                                              const ortc::Tensor<std::string>& input3,
                                              ortc::Tensor<std::string>& output) {
  // const OrtValue* inputs[4];
  // OrtTensorDimensions dims[4];

  auto& input = input1.Data();
  const int64_t* p_indices = input2.Data();
  int64_t size = input3.NumberOfElement();
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
  output.SetStringOutput(dense, shape_out);
}
