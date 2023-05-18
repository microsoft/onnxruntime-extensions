// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include <dlib/matrix.h>

template <bool with_norm = false>
struct STFT : public BaseKernel {
  STFT(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
    onesided_ = TryToGetAttributeWithDefault<int64_t>("onesided", 0LL);
  }

  void Compute(const ortc::Tensor<float>& x1,
               int64_t n_fft,
               int64_t hop_length,
               const ortc::Span<float>& x4,
               int64_t frame_length,
               ortc::Tensor<float>& output0) {
    const float* X = x1.Data();
    const float* window = x4.Data();
    const auto& dimensions = x1.Shape();
    const auto& win_dim = x4.Shape();

    if (dimensions.size() < 2 || dimensions.size() != dimensions[1]) {
      ORTX_CXX_API_THROW("[Stft] Only batch == 1 tensor supported.", ORT_INVALID_ARGUMENT);
    }
    if (frame_length != n_fft) {
      ORTX_CXX_API_THROW("[Stft] Only support size of FFT equals the frame length.", ORT_INVALID_ARGUMENT);
    }

    auto win_length = x4.size();
    dlib::matrix<float> dm_x = dlib::mat(X, 1, dimensions[1]);
    dlib::matrix<float> hann_win = dlib::mat(window, 1, win_length);

    auto m_stft = dlib::stft(
        dm_x, [&hann_win](size_t x, size_t len) { return hann_win(0, x); },
        n_fft, win_length, hop_length);

    if (onesided_) {
      m_stft = dlib::subm(m_stft, 0, 0, m_stft.nr(), (m_stft.nc() >> 1) + 1);
    }

    if (with_norm) {
      dlib::matrix<float> result = dlib::norm(m_stft);
      result = dlib::trans(result);
      std::vector<int64_t> outdim = {1, result.nr(), result.nc()};
      auto result_size = result.size();
      float* out0 = output0.Allocate(outdim);
      memcpy(out0, result.steal_memory().get(), result_size * sizeof(float));
    } else {
      auto result = m_stft;
      // No transpose here since it is done on copying data,
      // switch nr and nc, so the output dim willbe tranposed one.
      std::vector<int64_t> outdim = {1, result.nc(), result.nr(), 2};
      float* out0 = output0.Allocate(outdim);
      for (size_t c = 0; c < result.nc(); ++c) {
        for (size_t r = 0; r < result.nr(); ++r) {
          *out0 = result(r, c).real();
          *(out0 + 1) = result(r, c).imag();
          out0 += 2;
        }
      }
    }
  }

  int64_t onesided_{};
};
