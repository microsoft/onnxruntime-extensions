// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include <dlib/matrix.h>

#define _USE_MATH_DEFINES
#include "math.h"

struct StftNormal {
  StftNormal() = default;

  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    return OrtW::GetOpAttribute(info, "onesided", onesided_);
  }

  OrtxStatus Compute(const ortc::Tensor<float>& pcm, int64_t n_fft, int64_t hop_length,
                     const ortc::Span<float>& win, int64_t frame_length, ortc::Tensor<float>& output0) const {
    auto X = pcm.Data();
    auto window = win.data_;
    auto dimensions = pcm.Shape();
    auto win_length = win.size();

    if (dimensions.size() < 2 || pcm.NumberOfElement() != dimensions[1]) {
      return {kOrtxErrorInvalidArgument, "[Stft] Only batch == 1 tensor supported."};
    }
    if (frame_length != n_fft) {
      return {kOrtxErrorInvalidArgument, "[Stft] Only support size of FFT equals the frame length."};
    }

    dlib::matrix<float> dm_x = dlib::mat(X, 1, dimensions[1]);
    dlib::matrix<float> fft_win = dlib::mat(window, 1, win_length);

    auto m_stft =
      dlib::stft(dm_x, [&fft_win](size_t x, size_t len) { return fft_win(0, x); }, n_fft, win_length, hop_length);

    if (onesided_) {
      m_stft = dlib::subm(m_stft, 0, 0, m_stft.nr(), (m_stft.nc() >> 1) + 1);
    }

    dlib::matrix<float> result = dlib::norm(m_stft);
    result = dlib::trans(result);
    std::vector<int64_t> outdim{1, result.nr(), result.nc()};
    auto result_size = result.size();
    auto out0 = output0.Allocate(outdim);
    memcpy(out0, result.steal_memory().get(), result_size * sizeof(float));

    return {};
  }

 private:
  int64_t onesided_{1};
};

  static std::vector<float> hann_window(int N) {
    std::vector<float> window(N);

    for (int n = 0; n < N; ++n) {
      // Original formula introduces more rounding errors than the current implementation
      // window[n] = static_cast<float>(0.5 * (1 - std::cos(2 * M_PI * n / (N - 1))));
      double n_sin = std::sin(M_PI * n / N);
      window[n] = static_cast<float>(n_sin * n_sin);
    }

    return window;
  }

  // Symmetric Hann window: matches torch.hann_window(N, periodic=False).
  // Uses the classic cosine formula with denominator (N-1).
  static std::vector<float> hann_window_symmetric(int N) {
    std::vector<float> window(N);

    for (int n = 0; n < N; ++n) {
      window[n] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * n / (N - 1)));
    }

    return window;
  }