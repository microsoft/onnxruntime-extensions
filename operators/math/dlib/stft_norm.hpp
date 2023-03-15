// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include <dlib/matrix.h>


struct KernelStft : BaseKernel {
  KernelStft(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
    onesided_ = TryToGetAttributeWithDefault<int64_t>("onesided", 1);
  }

  void Compute(OrtKernelContext* context) {
    const OrtValue* input_x1 = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_x2 = ort_.KernelContext_GetInput(context, 1);
    const OrtValue* input_x3 = ort_.KernelContext_GetInput(context, 2);
    const OrtValue* input_x4 = ort_.KernelContext_GetInput(context, 3);
    const OrtValue* input_x5 = ort_.KernelContext_GetInput(context, 4);

    const float* X = ort_.GetTensorData<float>(input_x1);
    auto n_fft = *ort_.GetTensorData<int64_t>(input_x2);
    auto hop_length = *ort_.GetTensorData<int64_t>(input_x3);
    auto win_length = *ort_.GetTensorData<int64_t>(input_x4);
    auto window = ort_.GetTensorData<float>(input_x5);

    OrtTensorDimensions dimensions(ort_, input_x1);
    if (dimensions.size() != 1) {
      ORTX_CXX_API_THROW("[StftNorm] Only 1-d tensor supported.", ORT_INVALID_ARGUMENT);
    }

    dlib::matrix<float> dm_x = dlib::mat(X, 1, dimensions[0]);
    dlib::matrix<float> hann_win = dlib::mat(window, 1, win_length);

    auto m_stft = dlib::stft(
        dm_x, [&hann_win](size_t x, size_t len) { return hann_win(0, x); },
       n_fft, win_length, hop_length);

    if (onesided_) {
      m_stft = dlib::subm(m_stft, 0, 0, m_stft.nr(), (m_stft.nc() >> 1) + 1);
    }

    dlib::matrix<float> result = dlib::norm(m_stft);
    result = dlib::trans(result);

    int64_t outdim[] = {result.nr(), result.nc()};
    auto result_size = result.size();
    OrtValue* output0 = ort_.KernelContext_GetOutput(
        context, 0, outdim, 2);
    float* out0 = ort_.GetTensorMutableData<float>(output0);

    memcpy(out0, result.steal_memory().get(), result_size * sizeof(float));
  }

  private:
    int64_t onesided_;
};

struct CustomOpStftNorm : OrtW::CustomOpBase<CustomOpStftNorm, KernelStft> {
  const char* GetName() const {
    return "StftNorm";
  }

  size_t GetInputTypeCount() const {
    return 5;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 0 || index == 4) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};
