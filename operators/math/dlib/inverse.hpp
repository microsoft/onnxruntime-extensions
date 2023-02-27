// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <dlib/matrix.h>
#include "ocos.h"

struct KernelInverse : BaseKernel {
  KernelInverse(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const float* X = ort_.GetTensorData<float>(input_X);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);
    if (dimensions.size() != 2) {
      throw std::runtime_error("Only 2-d matrix supported.");
    }

    OrtValue* output0 = ort_.KernelContext_GetOutput(
        context, 0, dimensions.data(), dimensions.size());
    float* out0 = ort_.GetTensorMutableData<float>(output0);

    dlib::matrix<float> dm_x(dimensions[0], dimensions[1]);
    std::copy(X, X + dm_x.size(), dm_x.begin());
    dlib::matrix<float> dm = dlib::inv(dm_x);
    memcpy(out0, dm.steal_memory().get(), dm_x.size() * sizeof(float));
  }
};

struct CustomOpInverse : OrtW::CustomOpBase<CustomOpInverse, KernelInverse> {
  const char* GetName() const {
    return "Inverse";
  }

  size_t GetInputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};
