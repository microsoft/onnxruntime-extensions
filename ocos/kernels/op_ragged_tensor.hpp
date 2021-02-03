// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels.h"
#include "utils.h"

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

    if (in_dimension.size() == 1) {
      std::vector<int64_t> output_shape{Size(in_dimension)};
      OrtValue* values = ort_.KernelContext_GetOutput(context, 0, output_shape.data(), output_shape.size());
      T* p_values = ort_.GetTensorMutableData<T>(values);
      memcpy(p_values, p_in_tensor, output_shape[0] * sizeof(T));

      std::vector<int64_t> indices_shape{2};
      OrtValue* indices = ort_.KernelContext_GetOutput(context, 1, indices_shape.data(), indices_shape.size());
      int64_t* p_indices = ort_.GetTensorMutableData<int64_t>(indices);
      p_indices[0] = 0;
      p_indices[1] = output_shape[0];
      return;
    }

    if (in_dimension.size() == 2) {
      std::vector<int64_t> output_shape{Size(in_dimension)};
      OrtValue* values = ort_.KernelContext_GetOutput(context, 0, output_shape.data(), output_shape.size());
      T* p_values = ort_.GetTensorMutableData<T>(values);
      memcpy(p_values, p_in_tensor, output_shape[0] * sizeof(T));

      std::vector<int64_t> indices_shape{in_dimension[0] + 1};
      OrtValue* indices = ort_.KernelContext_GetOutput(context, 1, indices_shape.data(), indices_shape.size());
      int64_t* p_indices = ort_.GetTensorMutableData<int64_t>(indices);
      int64_t index = 0, i = 0;
      for (i = 0; i < in_dimension[0]; ++i, index += in_dimension[1]) {
        p_indices[i] = index;
      }
      p_indices[i] = index;
      return;
    }

    throw std::runtime_error(MakeString(
        "Input tensor must have one or two dimensions but has ", in_dimension.size(), "."));
  }
};

template <typename T>
struct CustomOpTensorToRaggedTensor : Ort::CustomOpBase<CustomOpTensorToRaggedTensor<T>, KernelTensorToRaggedTensor<T>> {
  size_t GetInputTypeCount() const { return 1; }
  size_t GetOutputTypeCount() const { return 2; }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    return GetTensorType<T>();
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    switch (index) {
      case 0:
        return GetTensorType<T>();
      case 1:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      default:
        throw std::runtime_error("Unexpected output index.");
    }
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
