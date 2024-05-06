// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#ifdef ORT_SWIFT_PACKAGE_MANAGER_BUILD
#include "onnxruntime/onnxruntime_c_api.h"
#include "onnxruntime/onnxruntime_cxx_api.h"
#else
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#endif
#undef ORT_API_MANUAL_INIT
#include "custom_op/onnxruntime_f16.h"

// #include "ocos.h"
//  #include "cublas_v2.h"
#include <cuda_runtime.h>

namespace contrib {

enum class Reduction : int {
  None = 0,
  Add = 1,
  Mul = 2,
  Min = 3,
  Max = 4,
};

template <typename T>
inline ONNXTensorElementDataType onnx_type();
template <>
inline ONNXTensorElementDataType onnx_type<float>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
template <>
inline ONNXTensorElementDataType onnx_type<Ort::Custom::MFloat16>() { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; }

/**
 * This kernel implementation the fusion of ConstantOfShape and ScatterND.
 * The implementation does not use OrtLiteCustom as the input shape (first input)
 * is expected to be on CPU whereas the other outputs are expected to be on CUDA.
 */
template <typename T>
struct ScatterNDOfShapeKernel {
  ScatterNDOfShapeKernel(const OrtApi& api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 private:
  void ComputeNoAtomic(cudaStream_t& stream, const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& indices_shape, T* output_data,
                       const int64_t* indices_data, const T* updates_data) const;

  Reduction reduction_;
  int maxThreadPerBlock_;
};

template <typename T>
struct ScatterNDOfShapeOp : Ort::CustomOpBase<ScatterNDOfShapeOp<T>, ScatterNDOfShapeKernel<T>> {
  typedef Ort::CustomOpBase<ScatterNDOfShapeOp<T>, ScatterNDOfShapeKernel<T>> parent_type;
  ScatterNDOfShapeOp() : parent_type() {}
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { return std::make_unique<ScatterNDOfShapeKernel<T>>(api, info).release(); }
  const char* GetName() const { return "ScatterNDOfShape"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }

  std::size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(std::size_t index) const { return index == 2 ? onnx_type<T>() : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(std::size_t index) const { return INPUT_OUTPUT_REQUIRED; }
  OrtMemType GetInputMemoryType(std::size_t index) const { return index == 0 ? OrtMemTypeCPUInput : OrtMemTypeDefault; }

  std::size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(std::size_t index) const { return onnx_type<T>(); }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(std::size_t index) const { return INPUT_OUTPUT_REQUIRED; }
};

}  // namespace contrib
