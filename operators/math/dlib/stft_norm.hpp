// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include <dlib/matrix.h>

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

std::cerr << "[DEBUG] PCM shape: ";
for (auto d : dimensions) std::cerr << d << " ";
std::cerr << "\n[DEBUG] PCM num elements: " << pcm.NumberOfElement() << "\n";
std::cerr << "[DEBUG] X ptr=" << X << "\n";
std::cerr << "[DEBUG] win_length=" << win_length << "\n";
std::cerr << "[DEBUG] n_fft=" << n_fft << " hop=" << hop_length << " frame=" << frame_length << "\n";

    if (dimensions.size() < 2 || pcm.NumberOfElement() != dimensions[1]) {
      return {kOrtxErrorInvalidArgument, "[Stft] Only batch == 1 tensor supported."};
    }
    if (frame_length != n_fft) {
      return {kOrtxErrorInvalidArgument, "[Stft] Only support size of FFT equals the frame length."};
    }

    std::cerr << "1" << std::endl;

    dlib::matrix<float> dm_x = dlib::mat(X, 1, dimensions[1]);
    dlib::matrix<float> fft_win = dlib::mat(window, 1, win_length);
    std::cerr << "2" << std::endl;

    auto m_stft =
      dlib::stft(dm_x, [&fft_win](size_t x, size_t len) { return fft_win(0, x); }, n_fft, win_length, hop_length);
    std::cerr << "3" << std::endl;

    if (onesided_) {
      m_stft = dlib::subm(m_stft, 0, 0, m_stft.nr(), (m_stft.nc() >> 1) + 1);
    }
    std::cerr << "4" << std::endl;

    dlib::matrix<float> result = dlib::norm(m_stft);
        std::cerr << "5" << std::endl;

    result = dlib::trans(result);
        std::cerr << "6" << std::endl;

    std::vector<int64_t> outdim{1, result.nr(), result.nc()};
            std::cerr << "7" << std::endl;

    auto result_size = result.size();
                std::cerr << "8" << std::endl;

    auto out0 = output0.Allocate(outdim);
                std::cerr << "9" << std::endl;

    memcpy(out0, result.steal_memory().get(), result_size * sizeof(float));
                std::cerr << "10" << std::endl;


                std::cerr << std::flush;

    return {};
  }

 private:
  int64_t onesided_{1};
};
