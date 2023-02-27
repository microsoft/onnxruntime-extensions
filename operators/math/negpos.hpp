// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

struct KernelNegPos : BaseKernel {
  KernelNegPos(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const float* X = ort_.GetTensorData<float>(input_X);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output0 = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    float* out0 = ort_.GetTensorMutableData<float>(output0);
    OrtValue* output1 = ort_.KernelContext_GetOutput(context, 1, dimensions.data(), dimensions.size());
    float* out1 = ort_.GetTensorMutableData<float>(output1);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output0);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    for (int64_t i = 0; i < size; i++) {
      if (X[i] > 0) {
        out0[i] = 0;
        out1[i] = X[i];
      } else {
        out0[i] = X[i];
        out1[i] = 0;
      }
    }
  }
};

struct CustomOpNegPos : OrtW::CustomOpBase<CustomOpNegPos, KernelNegPos> {
  const char* GetName() const {
    return "NegPos";
  }

  size_t GetInputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  size_t GetOutputTypeCount() const {
    return 2;
  }

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};
