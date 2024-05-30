// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "string_utils.h"
#include "scatter_nd_of_shape_impl.cuh"

namespace contrib {

template <typename T>
struct ScatterNDOfShape {
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    std::string value;
    OrtStatusPtr status = OrtW::GetOpAttribute(info, "reduction", value);
    if (status != nullptr)
      return status;

    if (value == "add")
      reduction_ = ScatterReduction::Add;
    else if (value == "mul")
      reduction_ = ScatterReduction::Mul;
    else if (value == "min")
      reduction_ = ScatterReduction::Min;
    else if (value == "max")
      reduction_ = ScatterReduction::Max;
    else
      ORTX_CXX_API_THROW("Unexpected reduction, only Add is implemented.", ORT_RUNTIME_EXCEPTION);

    return nullptr;
  }

  OrtStatusPtr Compute(Ort::Custom::CUDAKernelContext* ctx,
                       const ortc::Tensor<int64_t>& output_shape,
                       const ortc::Tensor<int64_t>& indices,
                       const ortc::Tensor<T>& updates,
                       ortc::Tensor<T>& output) const {
    auto& output_shape_shape = output_shape.Shape();
    auto& indices_shape = indices.Shape();
    auto& updates_shape = updates.Shape();

    if (output_shape_shape.size() != 1 || output_shape_shape[0] == 0) {
      ORTX_CXX_API_THROW("output shape must be a 1D tensor", ORT_RUNTIME_EXCEPTION);
    }
    if (indices_shape[indices_shape.size() - 1] != 1) {
      ORTX_CXX_API_THROW("last dimension of the indices tensor should be one", ORT_RUNTIME_EXCEPTION);
    }

    const int64_t* shape_data = output_shape.Data();  // CPU pointer
    const int64_t* indices_data = indices.Data();  // GPU pointer
    const T* updates_data = updates.Data();  // GPU pointer
    std::vector<int64_t> voutput_shape(shape_data, shape_data + output_shape_shape[0]);
    T* output_data = output.Allocate(voutput_shape);  // GPU pointer
    LaunchScatterNDOfShapeKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()),
                                    voutput_shape,
                                    indices_shape,
                                    indices_data,
                                    updates_data,
                                    output_data,
                                    reduction_);
    return nullptr;
  }

  static OrtMemType GetInputMemoryType(size_t input_index) {
    if (input_index == 0)  // shape
      return OrtMemType::OrtMemTypeCPUInput;
    return OrtMemType::OrtMemTypeDefault;
  }

  ScatterReduction reduction_;
};

}  // namespace contrib
