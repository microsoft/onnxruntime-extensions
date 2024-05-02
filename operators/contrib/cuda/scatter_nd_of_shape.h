// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
// #include "cublas_v2.h"
#include <cuda_runtime.h>

#ifdef ORT_SWIFT_PACKAGE_MANAGER_BUILD
#include "onnxruntime/onnxruntime_cxx_api.h"
#else
#include "onnxruntime_cxx_api.h"
#endif

namespace contrib {

enum class Reduction : int {
  None = 0,
  Add = 1,
  Mul = 2,
  Min = 3,
  Max = 4,
};

/**
 * This kernel implementation the fusion of ConstantOfShape and ScatterND.
 * The implementation does not use OrtLiteCustom as the input shape (first input)
 * is expected to be on CPU wheeras the other outputs are expected to be on CUDA.
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
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  const char* GetExecutionProviderType() const;

  std::size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(std::size_t index) const;
  OrtMemType GetInputMemoryType(std::size_t index) const;

  std::size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(std::size_t index) const;
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(std::size_t index) const;
};

}  // namespace contrib
