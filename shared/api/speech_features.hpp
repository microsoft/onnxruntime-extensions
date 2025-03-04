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
      } else if (key == "win_fn") {
        win_fn_ = std::get<std::string>(value);
        if (win_fn_ != "hann" && win_fn_ != "hamming") {
          return {kOrtxErrorInvalidArgument, "[AudioFeatures]: Invalid window type."};
        }
      } else if (key == "hann_win") {
        auto& win = std::get<std::vector<double>>(value);
        fft_win_.resize(win.size());
        std::transform(win.begin(), win.end(), fft_win_.begin(), [](double x) { return static_cast<float>(x); });
      } else if (key != "_comment") {
        return {kOrtxErrorInvalidArgument, "[AudioFeatures]: Invalid key in the JSON configuration."};
      }
    }

    if (fft_win_.empty()) {
      if (win_fn_ == "hamming") {
        fft_win_ = hamming_window(frame_length_);
      } else {  // default to hann
        fft_win_ = hann_window(frame_length_);
      }
    }
    return {};
  }

  OrtxStatus STFTNorm(const ortc::Tensor<float>& pcm, ortc::Tensor<float>& stft_norm) {
    return stft_norm_.Compute(pcm, n_fft_, hop_length_, {fft_win_.data(), fft_win_.size()}, n_fft_, stft_norm);
  }

  OrtxStatus SpeechLibSTFTNorm(const ortc::Tensor<float>& pcm, ortc::Tensor<float>& stft_norm) {
    const float preemphasis = 0.97f;
    // # Spec 1: SpeechLib cut remaining sample insufficient for a hop
    // n_batch = (wav.shape[0] - win_length) // hop_length + 1
    auto pcm_length = pcm.Shape()[1];
    auto n_batch = (pcm_length - frame_length_) / hop_length_ + 1;
    auto pcm_data = pcm.Data();
    dlib::matrix<float> dm_x = dlib::mat(pcm_data, 1, pcm_length);

    // # Here we don't use stride_tricks since the input array may not satisfy
    // # memory layout requirement and we need writeable output
    // # Here we only use list of views before copy to desination
    // # so it is more efficient than broadcasting
    // y_frames = np.array(
    //     [wav[_stride : _stride + win_length] for _stride in range(0, hop_length * n_batch, hop_length)],
    //     dtype=np.float32,
    // )

    // # Spec 2: SpeechLib applies preemphasis within each batch
    // y_frames_prev = np.roll(y_frames, 1, axis=1)
    // y_frames_prev[:, 0] = y_frames_prev[:, 1]
    // y_frames = (y_frames - preemphasis * y_frames_prev) * 32768
    // S = np.fft.rfft(fft_window * y_frames, n=n_fft, axis=1).astype(np.complex64)

    // Step 1: Create y_frames_prev by rolling each row right by 1 and adjusting the first element
    dlib::matrix<float> y_frames_prev(n_batch, frame_length_);
    for (long r = 0; r < n_batch; ++r) {
      for (long c = 0; c < frame_length_; ++c) {
        if (c == 0) {
          y_frames_prev(r, c) = dm_x(0, r * hop_length_ + frame_length_ - 1);
        } else {
          y_frames_prev(r, c) = dm_x(0, r * hop_length_ + c - 1);
        }
      }
        y_frames_prev(r, 0) = y_frames_prev(r, 1);
    }

    // Step 2: Apply pre-emphasis and scale by 32768
    dlib::matrix<float> y_processed(n_batch, frame_length_);
    for (long r = 0; r < n_batch; ++r) {
      for (long c = 0; c < frame_length_; ++c) {
        float sample = dm_x(0, r * hop_length_ + c);
        float rolled_sample = y_frames_prev(r, c);
        y_processed(r, c) = (sample - preemphasis * rolled_sample) * 32768.0f;
      }
    }

    // Step 3: Apply FFT window to each frame
    for (long r = 0; r < n_batch; ++r) {
      for (long c = 0; c < frame_length_; ++c) {
        y_processed(r, c) *= fft_win_[c];
      }
    }

    // Step 4: Compute Full FFT for each frame (complex output)
    // add extra column for simulating STFTNorm output shape
    dlib::matrix<std::complex<float>> S(n_batch + 1, n_fft_ / 2 + 1);  // Use n_fft_ columns instead of n_rfft

    for (long r = 0; r < n_batch; ++r) {
      dlib::matrix<float, 1, 0> frame = rowm(y_processed, r);
      dlib::matrix<float, 1, 0> padded_frame(1, n_fft_);
      padded_frame = 0;

      long copy_length = (std::min)(frame_length_, n_fft_);
      dlib::set_subm(padded_frame, 0, 0, 1, copy_length) = dlib::subm(frame, 0, 0, 1, copy_length);

      // Compute real FFT (output is column vector with n_fft_/2 +1 elements)
      dlib::matrix<std::complex<float>> fft_result = dlib::fftr(padded_frame);

      // Store the FFT coefficients (only non-redundant part)
      for (long c = 0; c <= n_fft_ / 2; ++c) {
        S(r, c) = fft_result(0, c);
      }
    }

    // Compute spectral power (squared magnitude)
    auto S_norm = dlib::norm(S);
    dlib::matrix<float> spec_power = dlib::trans(S_norm);

    std::vector<int64_t> outdim{1, spec_power.nr(), spec_power.nc()};
    auto result_size = spec_power.size();
    auto out0 = stft_norm.Allocate(outdim);
    memcpy(out0, spec_power.steal_memory().get(), result_size * sizeof(float));

    return {};
  }

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

  static std::vector<float> hamming_window(int N) {
    std::vector<float> window(N);

    for (int n = 0; n < N; ++n) {
      // Original formula introduces more rounding errors than the current implementation
      // window[n] = static_cast<float>(0.54 - 0.46 * std::cos(2 * M_PI * n / (N - 1)));
      double n_sin = std::sin(M_PI * n / (N - 1));
      window[n] = static_cast<float>(0.08 + 0.92 * n_sin * n_sin);
    }

    return window;
  }

 private:
  StftNormal stft_norm_;
  int64_t n_fft_{};
  int64_t hop_length_{};
  int64_t frame_length_{};
  std::vector<float> fft_win_;
  std::string win_fn_{"hann"};
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
      } else if (key == "feature_first") {
        feature_first_ = std::get<int64_t>(value);
      } else if (key == "no_padding") {
        no_padding_ = std::get<int64_t>(value);
      }
      else {
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
    int64_t n_fill_zero_col = stft_shape[1];  // if 8k, fill 4k - 8k hz with zeros
    int64_t additional_row = 0;
    if (mel_filters_.nc() > stft_shape[1]) {
      n_fill_zero_col = mel_filters_.nc() - stft_shape[1] - 1;
      additional_row = stft_shape[1] - 1;
    }
    dlib::matrix<float> magnitudes(stft_shape[1] + additional_row, stft_shape[2] - 1);
    for (int i = 0; i < magnitudes.nr(); ++i) {
      if (i < n_fill_zero_col) {
        std::copy(stft_norm.Data() + i * stft_shape[2], stft_norm.Data() + (i + 1) * stft_shape[2] - 1,
                  magnitudes.begin() + i * magnitudes.nc());
      } else {
        std::fill(magnitudes.begin() + i * magnitudes.nc(), magnitudes.begin() + (i + 1) * magnitudes.nc(), 0.0f);
      }
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

    float* buff{};
    if (no_padding_ == 1) {
      buff = logmel.Allocate({log_spec.nc(), log_spec.nr()});
      std::memcpy(buff, log_spec.begin(), log_spec.size() * sizeof(float));
    } else {
      std::vector<int64_t> shape = {mel_filters_.nr(), n_samples_ / hop_length_};
      buff = logmel.Allocate(shape);
      std::fill(buff, buff + logmel.NumberOfElement(), (log_spec_min + 4.0f) / 4.0f);
    }

    if (buff == nullptr) {
      return {kOrtxErrorOutOfMemory, "Failed to allocate memory for logmel tensor."};
    }

    if (feature_first_ == 1) {
      for (int i = 0; i < log_spec.nr(); ++i) {
        auto row_len = log_spec.nc() * i;
        std::copy(log_spec.begin() + i * log_spec.nc(), log_spec.begin() + (i + 1) * log_spec.nc(), buff + i * log_spec.nc());
      }
    } else {
      for (int i = 0; i < log_spec.nc(); ++i) {
        for (int j = 0; j < log_spec.nr(); ++j) {
          buff[i * log_spec.nr() + j] = log_spec(j, i);
        }
      }
    }

    return {};
  }

  // Function to compute the Mel filterbank
  static dlib::matrix<float> MelFilterBank(int n_fft, int n_mels,
                                           int sr = 16000, float min_mel = 0,
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
  int64_t feature_first_{1};
  int64_t no_padding_{};
};

class SpeechLibLogMel {
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
      } else if (key == "feature_first") {
        feature_first_ = std::get<int64_t>(value);
      } else if (key == "no_padding") {
        no_padding_ = std::get<int64_t>(value);
      }
      else {
        return {kOrtxErrorInvalidArgument, "[LogMel]: Invalid key in the JSON configuration."};
      }
    }

    n_samples_ = n_sr_ * chunk_size;
    mel_filters_ = MelFilterBank(n_fft, n_mel, n_sr_);
    return {};
  }

  OrtxStatus Compute(const ortc::Tensor<float>& spec_power, ortc::Tensor<float>& log_fbank) {
    // Compute the Mel spectrogram by following Python code
    /*
      fbank_power = np.clip(spec_power.dot(self._mel), 1.0, None)
      log_fbank = np.log(fbank_power).astype(np.float32)
    */
    assert(spec_power.Shape().size() == 3 && spec_power.Shape()[0] == 1);
    std::vector<int64_t> stft_shape = spec_power.Shape();
    int64_t n_fill_zero_col = stft_shape[1];  // if 8k, fill 4k - 8k hz with zeros
    int64_t additional_row = 0;
    if (mel_filters_.nc() > stft_shape[1]) {
      n_fill_zero_col = mel_filters_.nc() - stft_shape[1] - 1;
      additional_row = stft_shape[1] - 1;
    }
    dlib::matrix<float> magnitudes(stft_shape[1] + additional_row, stft_shape[2] - 1);
    for (int i = 0; i < magnitudes.nr(); ++i) {
      if (i < n_fill_zero_col) {
        std::copy(spec_power.Data() + i * stft_shape[2], spec_power.Data() + (i + 1) * stft_shape[2] - 1,
                  magnitudes.begin() + i * magnitudes.nc());
      } else {
        std::fill(magnitudes.begin() + i * magnitudes.nc(), magnitudes.begin() + (i + 1) * magnitudes.nc(), 0.0f);
      }
    }

    dlib::matrix<float> mel_spec = mel_filters_ * magnitudes;
    for (int i = 0; i < mel_spec.nr(); ++i) {
      for (int j = 0; j < mel_spec.nc(); ++j) {
        mel_spec(i, j) = std::max(1.0f, mel_spec(i, j));
      }
    }

    dlib::matrix<float> log_spec = dlib::log(mel_spec);
    float* buff = log_fbank.Allocate({log_spec.nc(), log_spec.nr()});
    std::memcpy(buff, log_spec.begin(), log_spec.size() * sizeof(float));
    if (buff == nullptr) {
      return {kOrtxErrorOutOfMemory, "Failed to allocate memory for logmel tensor."};
    }

    for (int i = 0; i < log_spec.nc(); ++i) {
      for (int j = 0; j < log_spec.nr(); ++j) {
        buff[i * log_spec.nr() + j] = log_spec(j, i);
      }
    }

    return {};
  }

  /*
  def speechlib_mel(sample_rate, n_fft, n_mels, fmin=None, fmax=None):
    """Create a Mel filter-bank the same as SpeechLib FbankFC.

    bank_width = int(n_fft // 2 + 1)
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0
    assert fmin >= 0, "fmin cannot be negtive"
    assert fmin < fmax <= sample_rate / 2, "fmax must be between (fmin, samplerate / 2]"

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f):
        return int((f * n_fft / sample_rate) + 0.5)

    # Spec 1: FFT bin range [f2bin(fmin) + 1, f2bin(fmax) - 1]
    klo = f2bin(fmin) + 1
    khi = f2bin(fmax)

    khi = max(khi, klo)

    # Spec 2: SpeechLib uses trianges in Mel space
    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)

    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(0, n_mels):
        left = m_centers[m]
        center = m_centers[m + 1]
        right = m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms

    return matrix
  */
  static dlib::matrix<float> MelFilterBank(int n_fft, int n_mels,
                                           int sr = 16000, float fmin = 0.0f,
                                           float fmax = 7689.0f) {
    // Validate input parameters
    assert(fmin >= 0.0f && "fmin cannot be negative");
    assert((fmin < fmax && fmax <= static_cast<float>(sr) / 2.0f) && "fmax must be between (fmin, samplerate/2]");

    // Helper functions
    auto mel = [](float f) {
      return 1127.0f * std::log(1.0f + f / 700.0f);
    };

    // Convert FFT bin to Mel scale
    auto bin2mel = [n_fft, sr](int fft_bin) {
      float f = (static_cast<float>(fft_bin) * sr) / n_fft;
      return 1127.0f * std::log(1.0f + f / 700.0f);
    };

    // Convert frequency to FFT bin
    auto f2bin = [n_fft, sr](float f) {
      return static_cast<int>((f * n_fft) / sr + 0.5f);
    };

    // Spec 1: FFT bin range [f2bin(fmin) + 1, f2bin(fmax) - 1]
    int klo = f2bin(fmin) + 1;
    int khi = f2bin(fmax);
    khi = std::max(khi, klo); // Ensure khi is at least klo

    // Spec 2: Mel scale range from fmin to fmax
    float mlo = mel(fmin);
    float mhi = mel(fmax);
    std::vector<float> m_centers(n_mels + 2);
    float step = (mhi - mlo) / (n_mels + 1);

    // Compute the mel centers
    for (int i = 0; i < n_mels + 2; ++i) {
      m_centers[i] = mlo + i * step;
    }
    float ms = (mhi - mlo) / (n_mels + 1);

    // Create the Mel filterbank matrix (n_mels x bank_width)
    int bank_width = n_fft / 2 + 1;
    dlib::matrix<float> matrix(n_mels, bank_width);
    for (int i = 0; i < matrix.nr(); ++i) {
      for (int j = 0; j < matrix.nc(); ++j) {
        matrix(i, j) = 0.0f;
      }
    }

    // Fill in the matrix with Mel weights
    for (int m = 0; m < n_mels; ++m) {
      float left = m_centers[m];
      float center = m_centers[m + 1];
      float right = m_centers[m + 2];
      for (int fft_bin = klo; fft_bin < khi; ++fft_bin) {
        if (fft_bin >= bank_width) {
          continue; // Ensure we don't exceed matrix columns
        }
        float mbin = bin2mel(fft_bin);
        if (mbin > left && mbin < right) {
          float diff = std::abs(center - mbin);
          matrix(m, fft_bin) = std::max(0.0f, 1.0f - diff / ms);
        }
      }
    }

    return matrix;
  }

 private:
  int64_t n_samples_ = {};  // sr * chunk_size
  int64_t hop_length_{};
  const int64_t n_sr_{16000};
  dlib::matrix<float> mel_filters_;
  int64_t feature_first_{1};
  int64_t no_padding_{};
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
                     ortc::Tensor<bool>& audio_attention_mask,
                     ortc::Tensor<int64_t>& embeded_size) {
    int64_t sr_val = sr.Data()[0];
    ortc::Tensor<float> stft_norm(&CppAllocator::Instance());
    SpeechFeatures stft_normal;
    stft_normal.Init(sr_val == 8000? stft_normal_8k_attrs_: stft_normal_attrs_);
    auto status = stft_normal.SpeechLibSTFTNorm(pcm, stft_norm);
    if (!status.IsOk()) {
      return status;
    }
    
    // Currently we only support 8k and 16k Hz sampling rate.
    if (sr_val != 8000 && sr_val != 16000){
      return {kOrtxErrorInvalidArgument, "Only 8k and 16k Hz target sampling rate is supported."};
    }

    SpeechLibLogMel logmel;
    // attributes already are verified in Init method
    logmel.Init(sr_val == 8000 ? logmel_8k_attrs_: logmel_attrs_);
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
    auto audio_frames = ts_logmel.Shape()[0];
    auto embedded_size_data = embeded_size.Allocate({1});
    embedded_size_data[0] = std::ceil(static_cast<float>(audio_frames) / audio_compression_rate_);

    constexpr int64_t feat_stride = 1;
    auto attention = audio_attention_mask.Allocate({audio_frames * feat_stride});
    std::memset(attention, 1, audio_frames * feat_stride * sizeof(bool));
    return status;
  }

  static OrtxStatus AlignOutputs(std::vector<TensorPtr>& audio_result);

 private:
  AttrDict logmel_attrs_;
  AttrDict stft_normal_attrs_;

  AttrDict logmel_8k_attrs_;
  AttrDict stft_normal_8k_attrs_;

  int64_t audio_compression_rate_{8};
  int64_t qformer_compression_rate_{1};
};

}  // namespace ort_extensions
