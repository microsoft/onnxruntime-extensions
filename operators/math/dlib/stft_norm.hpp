// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include <dlib/matrix.h>

struct KernelStft : BaseKernel {
  KernelStft(const OrtApi& api, const OrtKernelInfo& info, bool return_magnitude)
      : BaseKernel(api, info), return_magnitude_(return_magnitude) {
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
    auto window = ort_.GetTensorData<float>(input_x4);
    auto frame_length = *ort_.GetTensorData<int64_t>(input_x5);

    OrtTensorDimensions dimensions(ort_, input_x1);
    OrtTensorDimensions win_dim(ort_, input_x4);
    if (dimensions.size() < 2 || dimensions.Size() != dimensions[1]) {
      ORTX_CXX_API_THROW("[Stft] Only batch == 1 tensor supported.", ORT_INVALID_ARGUMENT);
    }
    if (win_dim.size() != 1) {
      ORTX_CXX_API_THROW("[Stft] Only 1-d hanning window supported.", ORT_INVALID_ARGUMENT);
    }
    if (frame_length != n_fft) {
      ORTX_CXX_API_THROW("[Stft] Only support size of FFT equals the frame length.", ORT_INVALID_ARGUMENT);
    }

    auto win_length = win_dim[0];
    dlib::matrix<float> dm_x = dlib::mat(X, 1, dimensions[1]);
    dlib::matrix<float> hann_win = dlib::mat(window, 1, win_length);

    auto m_stft = dlib::stft(
        dm_x, [&hann_win](size_t x, size_t len) { return hann_win(0, x); },
        n_fft, win_length, hop_length);

    if (onesided_) {
      m_stft = dlib::subm(m_stft, 0, 0, m_stft.nr(), (m_stft.nc() >> 1) + 1);
    }

    if (return_magnitude_) {
      dlib::matrix<float> result = dlib::norm(m_stft);
      result = dlib::trans(result);
      int64_t outdim[] = {1, result.nr(), result.nc()};
      auto result_size = result.size();
      OrtValue* output0 = ort_.KernelContext_GetOutput(
          context, 0, outdim, 3);
      float* out0 = ort_.GetTensorMutableData<float>(output0);
      memcpy(out0, result.steal_memory().get(), result_size * sizeof(float));
    } else {
      auto result = m_stft;
      // No transpose here since it is done on copying data,
      // switch nr and nc, so the output dim willbe tranposed one.
      int64_t outdim[] = {1, result.nc(), result.nr(), 2};
      OrtValue* output0 = ort_.KernelContext_GetOutput(
          context, 0, outdim, 4);
      float* out0 = ort_.GetTensorMutableData<float>(output0);
      for (size_t c = 0; c < result.nc(); ++c) {
        for (size_t r = 0; r < result.nr(); ++r) {
          *out0 = result(r, c).real();
          *(out0 + 1) = result(r, c).imag();
          out0 += 2;
        }
      }
    }
  }

 private:
  bool return_magnitude_;
  int64_t onesided_;
};

struct CustomOpStft : OrtW::CustomOpBase<CustomOpStft, KernelStft> {
  const char* GetName() const {
    return op_name_;
  }

  size_t GetInputTypeCount() const {
    return 5;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    // pcm and window are float
    if (index == 0 || index == 3) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo& info) const {
    return new KernelStft(api, info, with_norm_);
  };

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

 protected:
  bool with_norm_ = false;
  const char* op_name_ = "STFT";
};

struct CustomOpStftNorm : CustomOpStft {
 public:
  CustomOpStftNorm() {
    with_norm_ = true;
    op_name_ = "StftNorm";
  }
};
