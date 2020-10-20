// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"


struct KernelMatrixBandPart {
  KernelMatrixBandPart(OrtApi api)
      : api_(api),
        ort_(api_) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const float* X = ort_.GetTensorData<float>(input_X);

  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

struct KernelMatrixDiag {
  KernelMatrixDiag(OrtApi api)
      : api_(api),
        ort_(api_) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const int64_t* X = ort_.GetTensorData<int64_t>(input_X);

    //// Setup output
    //OrtTensorDimensions dimensions(ort_, input_X);

    //OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    //float* out = ort_.GetTensorMutableData<float>(output);

    //OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    //int64_t size = ort_.GetTensorShapeElementCount(output_info);
    //ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

struct CustomOpMatrixDiag : Ort::CustomOpBase<CustomOpMatrixDiag, KernelMatrixDiag> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    return new KernelMatrixDiag(api);
  };

  const char* GetName() const { return "MatrixDiag"; };

  static const OrtCustomOp* GetInstance() {
    static CustomOpMatrixDiag c_op_schema_obj;
    return &c_op_schema_obj;
  }

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

};

struct CustomOpMatrixBandPart : Ort::CustomOpBase<CustomOpMatrixBandPart, KernelMatrixBandPart> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    return new KernelMatrixBandPart(api);
  };

  const char* GetName() const { return "MatrixBandPart"; };

  static const OrtCustomOp* GetInstance() {
    static CustomOpMatrixBandPart c_op_schema_obj;
    return &c_op_schema_obj;
  }

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };
};
