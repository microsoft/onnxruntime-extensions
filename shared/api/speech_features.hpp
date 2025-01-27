// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <dlib/matrix.h>
#include <math/dlib/stft_norm.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ort_extensions {

class SpeechFeatures {
 public:
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "n_fft") {
        n_fft_ = std::get<int64_t>(value);
      } else if (key == "hop_length") {
        hop_length_ = std::get<int64_t>(value);
      } else if (key == "frame_length") {
        frame_length_ = std::get<int64_t>(value);
      } else if (key == "hann_win") {
        auto& win = std::get<std::vector<double>>(value);
        hann_win_.resize(win.size());
        std::transform(win.begin(), win.end(), hann_win_.begin(), [](double x) { return static_cast<float>(x); });
      } else if (key != "_comment") {
        return {kOrtxErrorInvalidArgument, "[AudioFeatures]: Invalid key in the JSON configuration."};
      }
    }

    if (hann_win_.empty()) {
      hann_win_ = hann_window(frame_length_);
    }
    return {};
  }

  OrtxStatus STFTNorm(const ortc::Tensor<float>& pcm, ortc::Tensor<float>& stft_norm) {
    return stft_norm_.Compute(pcm, n_fft_, hop_length_, {hann_win_.data(), hann_win_.size()}, frame_length_, stft_norm);
  }

  static std::vector<float> hann_window(int N) {
    std::vector<float> window(N);

    for (int n = 0; n < N; ++n) {
      // this formula leads to more rounding errors than the one below
      // window[n] = static_cast<float>(0.5 * (1 - std::cos(2 * M_PI * n / (N - 1))));
      double n_sin = std::sin(M_PI * n / N);
      window[n] = static_cast<float>(n_sin * n_sin);
    }

    return window;
  }

 private:
  StftNormal stft_norm_;
  int64_t n_fft_{};
  int64_t hop_length_{};
  int64_t frame_length_{};
  std::vector<float> hann_win_;
};

class LogMel {
 public:
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    int n_fft = 0;
    int n_mel = 0;
    int chunk_size = 0;
    for (const auto& [key, value] : attrs) {
      if (key == "hop_length") {
        hop_length_ = std::get<int64_t>(value);
      } else if (key == "n_fft") {
        n_fft = std::get<int64_t>(value);
      } else if (key == "n_mel") {
        n_mel = std::get<int64_t>(value);
      } else if (key == "chunk_size") {
        chunk_size = std::get<int64_t>(value);
      } else {
        return {kOrtxErrorInvalidArgument, "[LogMel]: Invalid key in the JSON configuration."};
      }
    }

    n_samples_ = n_sr_ * chunk_size;
    mel_filters_ = MelFilterBank(n_fft, n_mel, n_sr_);
    return {};
  }

  OrtxStatus Compute(const ortc::Tensor<float>& stft_norm, ortc::Tensor<float>& logmel) {
    // Compute the Mel spectrogram by following Python code
    /*
      magnitudes = stft_norm[:, :, :-1]
      mel_spec = self.mel_filters @ magnitudes
      log_spec = torch.clamp(mel_spec, min=1e-10).log10()
      spec_min = log_spec.max() - 8.0
      log_spec = torch.maximum(log_spec, spec_min)
      spec_shape = log_spec.shape
      padding_spec = torch.ones(spec_shape[0],
                                spec_shape[1],
                                self.n_samples // self.hop_length - spec_shape[2],
                                dtype=torch.float)
      padding_spec *= spec_min
      log_spec = torch.cat((log_spec, padding_spec), dim=2)
      log_spec = (log_spec + 4.0) / 4.0
      return log_spec
    */
    assert(stft_norm.Shape().size() == 3 && stft_norm.Shape()[0] == 1);
    std::vector<int64_t> stft_shape = stft_norm.Shape();
    dlib::matrix<float> magnitudes(stft_norm.Shape()[1], stft_norm.Shape()[2] - 1);
    for (int i = 0; i < magnitudes.nr(); ++i) {
      std::copy(stft_norm.Data() + i * stft_shape[2], stft_norm.Data() + (i + 1) * stft_shape[2] - 1,
                magnitudes.begin() + i * magnitudes.nc());
    }

    dlib::matrix<float> mel_spec = mel_filters_ * magnitudes;
    for (int i = 0; i < mel_spec.nr(); ++i) {
      for (int j = 0; j < mel_spec.nc(); ++j) {
        mel_spec(i, j) = std::max(1e-10f, mel_spec(i, j));
      }
    }

    dlib::matrix<float> log_spec = dlib::log10(mel_spec);
    float log_spec_min = dlib::max(log_spec) - 8.0f;
    for (int i = 0; i < log_spec.nr(); ++i) {
      for (int j = 0; j < log_spec.nc(); ++j) {
        float v = std::max(log_spec(i, j), log_spec_min);
        v = (v + 4.0f) / 4.0f;
        log_spec(i, j) = v;
      }
    }

    std::vector<int64_t> shape = {mel_filters_.nr(), n_samples_ / hop_length_};
    float* buff = logmel.Allocate(shape);
    std::fill(buff, buff + logmel.NumberOfElement(), (log_spec_min + 4.0f) / 4.0f);
    for (int i = 0; i < log_spec.nr(); ++i) {
      auto row_len = log_spec.nc() * i;
      std::copy(log_spec.begin() + i * log_spec.nc(), log_spec.begin() + (i + 1) * log_spec.nc(), buff + i * shape[1]);
    }

    return {};
  }

  // Function to compute the Mel filterbank
  static dlib::matrix<float> MelFilterBank(int n_fft, int n_mels, int sr = 16000, float min_mel = 0,
                                           float max_mel = 45.245640471924965) {
    // Initialize the filterbank matrix
    dlib::matrix<float> fbank(n_mels, n_fft / 2 + 1);
    memset(fbank.begin(), 0, fbank.size() * sizeof(float));

    // Compute the frequency bins for the DFT
    std::vector<float> freq_bins(n_fft / 2 + 1);
    for (int i = 0; i <= n_fft / 2; ++i) {
      freq_bins[i] = i * sr / static_cast<float>(n_fft);
    }

    // Compute the Mel scale frequencies
    std::vector<float> mel(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
      mel[i] = min_mel + i * (max_mel - min_mel) / (n_mels + 1);
    }

    // Fill in the linear scale
    float f_min = 0.0f;
    float f_sp = 200.0f / 3.0f;
    std::vector<float> freqs(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
      freqs[i] = f_min + f_sp * mel[i];
    }

    // Nonlinear scale
    float min_log_hz = 1000.0f;
    float min_log_mel = (min_log_hz - f_min) / f_sp;
    float logstep = log(6.4) / 27.0;

    for (int i = 0; i < n_mels + 2; ++i) {
      if (mel[i] >= min_log_mel) {
        freqs[i] = min_log_hz * exp(logstep * (mel[i] - min_log_mel));
      }
    }

    std::vector<float> mel_bins = freqs;
    std::vector<float> mel_spacing(n_mels + 1);
    for (int i = 0; i < n_mels + 1; ++i) {
      mel_spacing[i] = mel_bins[i + 1] - mel_bins[i];
    }

    // Compute the ramps
    std::vector<std::vector<float>> ramps(n_mels + 2, std::vector<float>(n_fft / 2 + 1));
    for (int i = 0; i < n_mels + 2; ++i) {
      for (int j = 0; j <= n_fft / 2; ++j) {
        ramps[i][j] = mel_bins[i] - freq_bins[j];
      }
    }

    for (int i = 0; i < n_mels; ++i) {
      for (int j = 0; j <= n_fft / 2; ++j) {
        float left = -ramps[i][j] / mel_spacing[i];
        float right = ramps[i + 2][j] / mel_spacing[i + 1];
        fbank(i, j) = std::max(0.0f, std::min(left, right));
      }
    }

    // Energy normalization
    for (int i = 0; i < n_mels; ++i) {
      float energy_norm = 2.0f / (mel_bins[i + 2] - mel_bins[i]);
      for (int j = 0; j <= n_fft / 2; ++j) {
        fbank(i, j) *= energy_norm;
      }
    }

    return fbank;
  }

 private:
  int64_t n_samples_ = {};  // sr * chunk_size
  int64_t hop_length_{};
  const int64_t n_sr_{16000};
  dlib::matrix<float> mel_filters_;
};

class Phi4AudioEmbed {
 public:
  Phi4AudioEmbed() = default;
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key.find("stft_normal/") == 0) {
        stft_normal_attrs_[key.substr(12)] = value;
      } else if (key.find("logmel/") == 0) {
        logmel_attrs_[key.substr(7)] = value;
      } else if (key.find("stft_normal_8k/") == 0) {
        stft_normal_8k_attrs_[key.substr(15)] = value;
      } else if (key.find("logmel_8k/") == 0) {
        logmel_8k_attrs_[key.substr(10)] = value;
      } else if (key == "audio_compression_rate") {
        audio_compression_rate_ = std::get<int64_t>(value);
      } else if (key == "qformer_compression_rate") {
        qformer_compression_rate_ = std::get<int64_t>(value);
      } else {
        return {kOrtxErrorInvalidArgument, "[Phi4AudioEmbed]: Invalid key in the JSON configuration."};
      }
    }

    SpeechFeatures stft_normal;
    OrtxStatus status = stft_normal.Init(stft_normal_attrs_);
    if (!status.IsOk()) {
      return status;
    }

    LogMel logmel;
    return logmel.Init(logmel_attrs_);
  }

  OrtxStatus Compute(const ortc::Tensor<float>& pcm,
                     const ortc::Tensor<int64_t>& sr,
                     ortc::Tensor<float>& ts_logmel,
                     ortc::Tensor<int64_t>& embeded_size) {

    int64_t sr_val = sr.Data()[0];
    ortc::Tensor<float> stft_norm(&CppAllocator::Instance());
    SpeechFeatures stft_normal;
    stft_normal.Init(sr_val == 8000? stft_normal_8k_attrs_: stft_normal_attrs_);
    auto status = stft_normal.STFTNorm(pcm, stft_norm);
    if (!status.IsOk()) {
      return status;
    }

    LogMel logmel;
    // already checked in Init
    logmel.Init(sr_val == 8000? logmel_8k_attrs_: logmel_attrs_);
    status = logmel.Compute(stft_norm, ts_logmel);
    if (!status.IsOk()) {
      return status;
    }

    /*
    def _compute_audio_embed_size(self, audio_frames):
        integer = audio_frames // self.compression_rate
        remainder = audio_frames % self.compression_rate

        result = integer if remainder == 0 else integer + 1

        integer = result // self.qformer_compression_rate
        remainder = result % self.qformer_compression_rate
        result = integer if remainder == 0 else integer + 1  # qformer compression

        return result    
    */
    auto embedded_size_data = embeded_size.Allocate({1});
    embedded_size_data[0] = std::ceil(static_cast<float>(ts_logmel.Shape()[1]) / audio_compression_rate_);
    return status;
  }

 private:
  AttrDict logmel_attrs_;
  AttrDict stft_normal_attrs_;

  AttrDict logmel_8k_attrs_;
  AttrDict stft_normal_8k_attrs_;

  int64_t audio_compression_rate_{8};
  int64_t qformer_compression_rate_{1};
};

}  // namespace ort_extensions
