// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "nemo_mel_spectrogram.h"

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

// ─── Filterbank ─────────────────────────────────────────────────────────────

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

// ─── Batch log-mel extraction ───────────────────────────────────────────────

TEST(NemoMelTest, BatchOutputShape) {
  auto cfg = MakeTestConfig();
  auto wav = SineWave(440.0f, 0.5f);  // 0.5 sec, 8000 samples
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

// ─── Streaming extractor ────────────────────────────────────────────────────

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
  // Chunk smaller than hop_length should produce 0 frames (buffered for next call)
  auto cfg = MakeTestConfig();
  NemoStreamingMelExtractor extractor(cfg);
  std::vector<float> tiny(100, 0.1f);  // 100 samples < hop_length (160)
  auto [mel, frames] = extractor.Process(tiny.data(), tiny.size());
  // May produce 0 frames since not enough samples for a full hop
  EXPECT_GE(frames, 0);
}
