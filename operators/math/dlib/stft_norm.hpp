// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include <dlib/matrix.h>

struct STFT : public BaseKernel {
  STFT(const OrtApi& api, const OrtKernelInfo& info,
       bool with_norm = false) : BaseKernel(api, info),
                                 with_norm_(with_norm) {
    onesided_ = TryToGetAttributeWithDefault<int64_t>("onesided", 1);
  }

  void Compute(const ortc::Tensor<float>& input0,
               int64_t n_fft,
               int64_t hop_length,
               const ortc::Span<float>& input3,
               int64_t frame_length,
               ortc::Tensor<float>& output0) {
    auto X = input0.Data();
    auto window = input3.data();
    auto dimensions = input0.Shape();
    auto win_length = input3.size();

    if (dimensions.size() < 2 || input0.NumberOfElement() != dimensions[1]) {
      ORTX_CXX_API_THROW("[Stft] Only batch == 1 tensor supported.", ORT_INVALID_ARGUMENT);
    }
    if (frame_length != n_fft) {
      ORTX_CXX_API_THROW("[Stft] Only support size of FFT equals the frame length.", ORT_INVALID_ARGUMENT);
    }

    dlib::matrix<float> dm_x = dlib::mat(X, 1, dimensions[1]);
    dlib::matrix<float> hann_win = dlib::mat(window, 1, win_length);

    auto m_stft = dlib::stft(
        dm_x, [&hann_win](size_t x, size_t len) { return hann_win(0, x); },
        n_fft, win_length, hop_length);

    if (onesided_) {
      m_stft = dlib::subm(m_stft, 0, 0, m_stft.nr(), (m_stft.nc() >> 1) + 1);
    }

    if (with_norm_) {
      dlib::matrix<float> result = dlib::norm(m_stft);
      result = dlib::trans(result);
      std::vector<int64_t> outdim{1, result.nr(), result.nc()};
      auto result_size = result.size();
      auto out0 = output0.Allocate(outdim);
      memcpy(out0, result.steal_memory().get(), result_size * sizeof(float));
    } else {
      auto result = m_stft;
      // No transpose here since it is done on copying data,
      // switch nr and nc, so the output dim willbe tranposed one.
      std::vector<int64_t> outdim{1, result.nc(), result.nr(), 2};
      auto out0 = output0.Allocate(outdim);
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
  int64_t onesided_{};
  bool with_norm_{};
};

struct StftNormal : public STFT {
  StftNormal(const OrtApi& api, const OrtKernelInfo& info) : STFT(api, info, true) {}
  void Compute(const ortc::Tensor<float>& input0,
               int64_t n_fft,
               int64_t hop_length,
               const ortc::Span<float>& input3,
               int64_t frame_length,
               ortc::Tensor<float>& output0) {
    STFT::Compute(input0, n_fft, hop_length, input3, frame_length, output0);
  }
};
