// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels.h"
#include "utils.h"
#include "sparse.hpp"

struct KernelRaggedTensorToSparse : BaseKernel {
  KernelRaggedTensorToSparse(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpRaggedTensorToSparse : Ort::CustomOpBase<CustomOpRaggedTensorToSparse, KernelRaggedTensorToSparse> {
  size_t GetInputTypeCount() const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  const char* GetName() const;
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
};

template <typename T>
struct KernelTensorToRaggedTensor : BaseKernel {
  KernelTensorToRaggedTensor(OrtApi api) : BaseKernel(api) {}
  void Compute(OrtKernelContext* context) {
    const OrtValue* in_tensor = ort_.KernelContext_GetInput(context, 0);
    const T* p_in_tensor = ort_.GetTensorData<T>(in_tensor);

    OrtTensorDimensions in_dimension(ort_, in_tensor);
    SparseInTensor<T> ragged;

    if (in_dimension.size() == 1) {
      std::vector<int64_t> in_dimension2(2);
      in_dimension2[0] = in_dimension[0];
      in_dimension2[1] = 1;
      SparseInTensor<T>::create_ragged_from_dense(in_dimension2, p_in_tensor, ragged);
    } else if (in_dimension.size() == 2) {
      SparseInTensor<T>::create_ragged_from_dense(in_dimension, p_in_tensor, ragged);
    } else {
      throw std::runtime_error(MakeString(
          "Input tensor must have one or two dimensions but has ", in_dimension.size(), "."));
    }
    std::vector<int64_t> output_shape{ragged.size()};
    OrtValue* values = ort_.KernelContext_GetOutput(
        context, 0, output_shape.data(), output_shape.size());
    uint8_t* p_values = ort_.GetTensorMutableData<uint8_t>(values);
    memcpy(p_values, ragged.buffer(), ragged.size());
    return;
  }
};

template <typename T>
struct CustomOpTensorToRaggedTensor : Ort::CustomOpBase<CustomOpTensorToRaggedTensor<T>, KernelTensorToRaggedTensor<T>> {
  size_t GetInputTypeCount() const { return 1; }
  size_t GetOutputTypeCount() const { return 1; }

  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return GetTensorType<T>();
  }

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  }

  const char* GetName() const {
    switch (GetTensorType<T>()) {
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return "TensorToRaggedTensorFloat";
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return "TensorToRaggedTensorDouble";
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return "TensorToRaggedTensorInt64";
      default:
        throw std::runtime_error("Not implemented for this type.");
    }
  }

  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
    return new KernelTensorToRaggedTensor<T>(api);
  }
};
