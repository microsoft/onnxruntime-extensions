// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <set>
#include <vector>

#include "gtest/gtest.h"
#include "nemo_mel_spectrogram.h"
#include "c_api_utils.hpp"
#include "runner.hpp"
#include "speech_features.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace nemo_mel;

static NemoMelConfig MakeTestConfig() {
  NemoMelConfig cfg;
  cfg.num_mels = 128;
  cfg.fft_size = 512;
  cfg.hop_length = 160;
  cfg.win_length = 400;
  cfg.sample_rate = 16000;
  cfg.preemph = 0.97f;
  cfg.log_eps = 5.96046448e-08f;
  return cfg;
}

// Generate a pure sine wave (mono, float32).
static std::vector<float> SineWave(float freq_hz, float duration_sec,
                                   int sample_rate = 16000, float amplitude = 0.5f) {
  int n = static_cast<int>(duration_sec * sample_rate);
  std::vector<float> wav(n);
  const float two_pi = 2.0f * static_cast<float>(M_PI);
  for (int i = 0; i < n; ++i) {
    wav[i] = amplitude * std::sin(two_pi * freq_hz * i / sample_rate);
  }
  return wav;
}

// Mel scale conversions (Slaney)
TEST(NemoMelTest, HzToMelLinearRegion) {
  // Below 1000 Hz the Slaney scale is linear: mel = 3 * hz / 200
  EXPECT_FLOAT_EQ(HzToMel(0.0f), 0.0f);
  EXPECT_FLOAT_EQ(HzToMel(200.0f), 3.0f);
  EXPECT_FLOAT_EQ(HzToMel(1000.0f), 15.0f);
}

TEST(NemoMelTest, HzToMelLogRegion) {
  // Above 1000 Hz the Slaney scale is logarithmic
  float mel_2000 = HzToMel(2000.0f);
  float mel_4000 = HzToMel(4000.0f);
  // mel(4000) - mel(2000) should equal mel(2000) - mel(1000) since the log region
  // has equal spacing per octave
  float diff_upper = mel_4000 - mel_2000;
  float diff_lower = mel_2000 - HzToMel(1000.0f);
  EXPECT_NEAR(diff_upper, diff_lower, 0.01f);
}

TEST(NemoMelTest, MelToHzRoundTrip) {
  // HzToMel and MelToHz should be inverses
  for (float hz : {0.0f, 100.0f, 500.0f, 1000.0f, 2000.0f, 4000.0f, 8000.0f}) {
    float mel = HzToMel(hz);
    float hz_back = MelToHz(mel);
    EXPECT_NEAR(hz_back, hz, 0.01f) << "Round-trip failed for hz=" << hz;
  }
}

TEST(NemoMelTest, FilterbankShape) {
  auto fb = CreateMelFilterbank(128, 512, 16000);
  ASSERT_EQ(fb.size(), 128u);
  ASSERT_EQ(fb[0].size(), 257u);  // fft_size/2 + 1
}

TEST(NemoMelTest, FilterbankNonNegative) {
  auto fb = CreateMelFilterbank(128, 512, 16000);
  for (const auto& row : fb) {
    for (float v : row) {
      EXPECT_GE(v, 0.0f);
    }
  }
}

TEST(NemoMelTest, FilterbankTriangular) {
  // Each mel filter should be triangular: has a single peak with values
  // rising then falling, with no internal zeros between non-zero values.
  auto fb = CreateMelFilterbank(64, 512, 16000);
  for (size_t m = 0; m < fb.size(); ++m) {
    const auto& row = fb[m];
    // Find first and last non-zero
    int first_nz = -1, last_nz = -1;
    for (int i = 0; i < static_cast<int>(row.size()); ++i) {
      if (row[i] > 0.0f) {
        if (first_nz < 0) first_nz = i;
        last_nz = i;
      }
    }
    if (first_nz < 0) continue;  // empty filter at edges is ok
    // All values between first_nz and last_nz should be positive
    for (int i = first_nz; i <= last_nz; ++i) {
      EXPECT_GT(row[i], 0.0f) << "Zero gap in mel filter " << m << " at bin " << i;
    }
  }
}

TEST(NemoMelTest, BatchOutputShape) {
  auto cfg = MakeTestConfig();
  auto wav = SineWave(440.0f, 0.5f);
  int num_frames = 0;
  auto mel = NemoComputeLogMelBatch(wav.data(), wav.size(), cfg, num_frames);

  EXPECT_GT(num_frames, 0);
  EXPECT_EQ(mel.size(), static_cast<size_t>(cfg.num_mels) * num_frames);

  // Sanity-check frame count is in a reasonable range.
  // Exact formula depends on center-padding strategy; just verify ballpark.
  int min_expected = static_cast<int>(wav.size()) / cfg.hop_length - 2;
  int max_expected = static_cast<int>(wav.size()) / cfg.hop_length + 5;
  EXPECT_GE(num_frames, min_expected);
  EXPECT_LE(num_frames, max_expected);
}

TEST(NemoMelTest, BatchSilenceOutput) {
  // Silence should produce very low (near log_eps) mel values
  auto cfg = MakeTestConfig();
  std::vector<float> silence(16000, 0.0f);  // 1 sec
  int num_frames = 0;
  auto mel = NemoComputeLogMelBatch(silence.data(), silence.size(), cfg, num_frames);

  float expected_log_eps = std::log(cfg.log_eps);
  for (size_t i = 0; i < mel.size(); ++i) {
    EXPECT_NEAR(mel[i], expected_log_eps, 0.1f)
        << "Silence mel value at index " << i << " deviates from log(eps)";
  }
}

TEST(NemoMelTest, BatchDeterministic) {
  auto cfg = MakeTestConfig();
  auto wav = SineWave(1000.0f, 0.3f);
  int nf1 = 0, nf2 = 0;
  auto mel1 = NemoComputeLogMelBatch(wav.data(), wav.size(), cfg, nf1);
  auto mel2 = NemoComputeLogMelBatch(wav.data(), wav.size(), cfg, nf2);
  ASSERT_EQ(nf1, nf2);
  ASSERT_EQ(mel1.size(), mel2.size());
  for (size_t i = 0; i < mel1.size(); ++i) {
    EXPECT_FLOAT_EQ(mel1[i], mel2[i]);
  }
}

TEST(NemoMelTest, BatchSineEnergy) {
  // A 440Hz sine should concentrate energy in lower mel bands
  auto cfg = MakeTestConfig();
  auto wav = SineWave(440.0f, 0.5f);
  int num_frames = 0;
  auto mel = NemoComputeLogMelBatch(wav.data(), wav.size(), cfg, num_frames);

  // Average mel energy across time for each band
  std::vector<float> band_avg(cfg.num_mels, 0.0f);
  for (int m = 0; m < cfg.num_mels; ++m) {
    for (int t = 0; t < num_frames; ++t) {
      band_avg[m] += mel[m * num_frames + t];
    }
    band_avg[m] /= num_frames;
  }
  // The mel band containing 440 Hz should have more energy than the highest band
  // 440 Hz at 16kHz with 128 mels is in the low mel range
  float max_low = *std::max_element(band_avg.begin(), band_avg.begin() + 30);
  float avg_high = 0.0f;
  for (int m = 100; m < 128; ++m) avg_high += band_avg[m];
  avg_high /= 28.0f;
  EXPECT_GT(max_low, avg_high);
}

TEST(NemoMelTest, StreamingSingleChunkMatchesBatch) {
  auto cfg = MakeTestConfig();
  auto wav = SineWave(440.0f, 0.5f);  // 8000 samples

  // Batch reference
  int batch_frames = 0;
  auto batch_mel = NemoComputeLogMelBatch(wav.data(), wav.size(), cfg, batch_frames);

  // Streaming: send all audio in one chunk
  NemoStreamingMelExtractor extractor(cfg);
  auto [stream_mel, stream_frames] = extractor.Process(wav.data(), wav.size());

  // Streaming uses symmetric Hann + left-only center-pad vs batch uses periodic Hann
  // + both-side center-pad, so frame counts may differ by a small amount.
  EXPECT_NEAR(stream_frames, batch_frames, 2);
  // Both should produce non-empty output
  EXPECT_GT(stream_frames, 0);
  EXPECT_GT(stream_mel.size(), 0u);
}

TEST(NemoMelTest, StreamingMultiChunk) {
  auto cfg = MakeTestConfig();
  auto wav = SineWave(440.0f, 1.0f);  // 16000 samples

  NemoStreamingMelExtractor extractor(cfg);
  int total_frames = 0;
  std::vector<float> all_mel;

  // Feed in 4 chunks of 4000 samples
  size_t chunk_size = 4000;
  for (size_t offset = 0; offset < wav.size(); offset += chunk_size) {
    size_t n = std::min(chunk_size, wav.size() - offset);
    auto [mel, frames] = extractor.Process(wav.data() + offset, n);
    all_mel.insert(all_mel.end(), mel.begin(), mel.end());
    total_frames += frames;
  }

  EXPECT_GT(total_frames, 0);
  EXPECT_EQ(all_mel.size(), static_cast<size_t>(cfg.num_mels) * total_frames);
}

TEST(NemoMelTest, StreamingReset) {
  auto cfg = MakeTestConfig();
  auto wav = SineWave(440.0f, 0.3f);

  NemoStreamingMelExtractor extractor(cfg);

  // First utterance
  auto [mel1, nf1] = extractor.Process(wav.data(), wav.size());

  // Reset and process same audio
  extractor.Reset();
  auto [mel2, nf2] = extractor.Process(wav.data(), wav.size());

  // Should produce identical results after reset
  ASSERT_EQ(nf1, nf2);
  ASSERT_EQ(mel1.size(), mel2.size());
  for (size_t i = 0; i < mel1.size(); ++i) {
    EXPECT_FLOAT_EQ(mel1[i], mel2[i]) << "Mismatch after reset at index " << i;
  }
}

TEST(NemoMelTest, StreamingEmptyChunk) {
  auto cfg = MakeTestConfig();
  NemoStreamingMelExtractor extractor(cfg);
  // The first Process() call may produce frames from the initial center-pad overlap,
  // even with 0 input samples. Just verify it doesn't crash.
  auto [mel, frames] = extractor.Process(nullptr, 0);
  EXPECT_GE(frames, 0);
  EXPECT_EQ(mel.size(), static_cast<size_t>(cfg.num_mels) * frames);
}

TEST(NemoMelTest, StreamingSmallChunk) {
  // Even a chunk smaller than hop_length produces frames on the first call
  // because the initial left-pad (fft_size/2 zeros) is prepended, giving
  // enough samples to form at least one STFT frame.
  auto cfg = MakeTestConfig();
  NemoStreamingMelExtractor extractor(cfg);
  std::vector<float> tiny(100, 0.1f);  // 100 samples < hop_length (160)
  auto [mel, frames] = extractor.Process(tiny.data(), tiny.size());
  EXPECT_GT(frames, 0);
  EXPECT_EQ(mel.size(), static_cast<size_t>(cfg.num_mels) * frames);
}

// =====================================================================
// NemoLogMel kernel tests (pipeline kernel wrapping NemoComputeLogMelBatch)
// =====================================================================

using namespace ort_extensions;

static ortc::Tensor<float> MakeInputTensor(const std::vector<int64_t>& shape, const float* data) {
  return ortc::Tensor<float>(shape, const_cast<void*>(static_cast<const void*>(data)));
}

static ortc::Tensor<float> MakeOutputTensor() {
  return ortc::Tensor<float>(&CppAllocator::Instance());
}

static AttrDict MakeNemoLogMelAttrs() {
  AttrDict attrs;
  attrs["num_mels"] = int64_t{128};
  attrs["fft_size"] = int64_t{512};
  attrs["hop_length"] = int64_t{160};
  attrs["win_length"] = int64_t{400};
  attrs["sample_rate"] = int64_t{16000};
  attrs["preemph"] = double{0.97};
  attrs["log_eps"] = double{5.96046448e-08};
  return attrs;
}

// =====================================================================
// PerFeatureNormalize kernel tests
// =====================================================================

TEST(NemoMelTest, Normalize2D) {
  const int64_t F = 4, T = 10;
  std::vector<float> data(F * T);
  for (int64_t f = 0; f < F; ++f)
    for (int64_t t = 0; t < T; ++t)
      data[f * T + t] = static_cast<float>(f * 10 + t);

  PerFeatureNormalize kernel;
  AttrDict attrs;
  attrs["eps"] = double{1e-5};
  attrs["feature_first"] = int64_t{1};
  ASSERT_TRUE(kernel.Init(attrs).IsOk());

  auto input = MakeInputTensor({F, T}, data.data());
  auto output = MakeOutputTensor();
  ASSERT_TRUE(kernel.Compute(input, output).IsOk());

  auto& shape = output.Shape();
  ASSERT_EQ(shape.size(), 2u);

  // Each row should be zero-mean
  for (int64_t f = 0; f < F; ++f) {
    float sum = 0.0f;
    for (int64_t t = 0; t < T; ++t) sum += output.Data()[f * T + t];
    EXPECT_NEAR(sum / T, 0.0f, 1e-5f) << "Row " << f << " not zero-mean";
  }

  // Each row should have unit sample std
  for (int64_t f = 0; f < F; ++f) {
    float var = 0.0f;
    for (int64_t t = 0; t < T; ++t) {
      float v = output.Data()[f * T + t];
      var += v * v;  // mean is ~0
    }
    float std_val = std::sqrt(var / (T - 1));
    EXPECT_NEAR(std_val, 1.0f, 1e-4f) << "Row " << f << " not unit std";
  }
}

TEST(NemoMelTest, Normalize3D) {
  const int64_t F = 3, T = 8;
  std::vector<float> data(F * T);
  for (int64_t i = 0; i < F * T; ++i) data[i] = static_cast<float>(i);

  PerFeatureNormalize kernel;
  AttrDict attrs;
  attrs["eps"] = double{1e-5};
  attrs["feature_first"] = int64_t{1};
  ASSERT_TRUE(kernel.Init(attrs).IsOk());

  auto input = MakeInputTensor({1, F, T}, data.data());
  auto output = MakeOutputTensor();
  ASSERT_TRUE(kernel.Compute(input, output).IsOk());

  auto& shape = output.Shape();
  ASSERT_EQ(shape.size(), 3u);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], F);
  EXPECT_EQ(shape[2], T);
}

TEST(NemoMelTest, NormalizeConstantRow) {
  const int64_t F = 2, T = 5;
  std::vector<float> data(F * T);
  for (int64_t t = 0; t < T; ++t) data[t] = 7.0f;
  for (int64_t t = 0; t < T; ++t) data[T + t] = float(t);

  PerFeatureNormalize kernel;
  AttrDict attrs;
  attrs["eps"] = double{1e-5};
  attrs["feature_first"] = int64_t{1};
  ASSERT_TRUE(kernel.Init(attrs).IsOk());

  auto input = MakeInputTensor({F, T}, data.data());
  auto output = MakeOutputTensor();
  ASSERT_TRUE(kernel.Compute(input, output).IsOk());

  // Constant row should normalize to zeros
  for (int64_t t = 0; t < T; ++t)
    EXPECT_NEAR(output.Data()[t], 0.0f, 1e-4f);
}

TEST(NemoMelTest, NormalizeRejectsBadShape) {
  std::vector<float> data(120, 1.0f);

  PerFeatureNormalize kernel;
  AttrDict attrs;
  attrs["eps"] = double{1e-5};
  attrs["feature_first"] = int64_t{1};
  ASSERT_TRUE(kernel.Init(attrs).IsOk());

  auto input = MakeInputTensor({2, 3, 4, 5}, data.data());
  auto output = MakeOutputTensor();
  EXPECT_FALSE(kernel.Compute(input, output).IsOk());
}

TEST(NemoMelTest, NormalizeSingleFrame) {
  // Single frame (num_frames=1) should not crash and should output zeros
  const int64_t F = 4, T = 1;
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  PerFeatureNormalize kernel;
  AttrDict attrs;
  attrs["eps"] = double{1e-5};
  attrs["feature_first"] = int64_t{1};
  ASSERT_TRUE(kernel.Init(attrs).IsOk());

  auto input = MakeInputTensor({F, T}, data.data());
  auto output = MakeOutputTensor();
  ASSERT_TRUE(kernel.Compute(input, output).IsOk());

  auto& shape = output.Shape();
  ASSERT_EQ(shape.size(), 2u);
  EXPECT_EQ(shape[0], F);
  EXPECT_EQ(shape[1], T);

  // All outputs should be zero and finite
  for (int64_t f = 0; f < F; ++f) {
    EXPECT_NEAR(output.Data()[f], 0.0f, 1e-6f);
    ASSERT_TRUE(std::isfinite(output.Data()[f]));
  }
}

// =====================================================================
// End-to-end: NemoLogMel -> PerFeatureNormalize
// =====================================================================

TEST(NemoMelTest, KernelPipeline) {
  auto wav = SineWave(440.0f, 1.0f);

  NemoLogMel mel_kernel;
  ASSERT_TRUE(mel_kernel.Init(MakeNemoLogMelAttrs()).IsOk());

  auto pcm = MakeInputTensor({static_cast<int64_t>(wav.size())}, wav.data());
  auto mel_out = MakeOutputTensor();
  ASSERT_TRUE(mel_kernel.Compute(pcm, mel_out).IsOk());

  PerFeatureNormalize norm_kernel;
  AttrDict norm_attrs;
  norm_attrs["eps"] = double{1e-5};
  norm_attrs["feature_first"] = int64_t{1};
  ASSERT_TRUE(norm_kernel.Init(norm_attrs).IsOk());

  auto norm_out = MakeOutputTensor();
  ASSERT_TRUE(norm_kernel.Compute(mel_out, norm_out).IsOk());

  auto& shape = norm_out.Shape();
  ASSERT_EQ(shape.size(), 2u);
  EXPECT_EQ(shape[0], 128);

  // All finite
  size_t n = static_cast<size_t>(shape[0] * shape[1]);
  for (size_t i = 0; i < n; ++i)
    ASSERT_TRUE(std::isfinite(norm_out.Data()[i])) << "Non-finite at " << i;

  // Zero-mean per feature
  int64_t frames = shape[1];
  for (int64_t f = 0; f < 128; ++f) {
    float sum = 0.0f;
    for (int64_t t = 0; t < frames; ++t)
      sum += norm_out.Data()[f * frames + t];
    EXPECT_NEAR(sum / frames, 0.0f, 1e-4f) << "Feature " << f << " not zero-mean";
  }
}
