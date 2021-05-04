// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <dlib/matrix.h>
#include "ocos.h"


struct KernelInverse : BaseKernel {
  KernelInverse(OrtApi api) : BaseKernel(api) {
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

    OrtValue* output0 = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    float* out0 = ort_.GetTensorMutableData<float>(output0);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output0);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    dlib::matrix<float> dm(dimensions[0], dimensions[1]);
    // Do computation
    for (int64_t i = 0; i < size; i++) {
      out0[i] = dm(i / dimensions[1], i % dimensions[1]);
    }
  }
};

struct CustomOpInverse : Ort::CustomOpBase<CustomOpInverse, KernelInverse> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
    return new KernelInverse(api);
  }

  const char* GetName() const {
    return "Inverse";
  }

  size_t GetInputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};
