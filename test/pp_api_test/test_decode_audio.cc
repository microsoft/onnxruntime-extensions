// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "ortx_cpp_helper.h"
#include "ortx_extractor.h"

// Helper: generate a 16-bit PCM WAV buffer from a sine wave.
static std::vector<uint8_t> MakeSineWav(float freq_hz, float duration_sec, int sample_rate = 16000) {
  int n = static_cast<int>(duration_sec * sample_rate);
  size_t data_size = n * 2;
  size_t file_size = 44 + data_size;
  std::vector<uint8_t> wav(file_size);
  uint8_t* p = wav.data();

  std::memcpy(p, "RIFF", 4); p += 4;
  uint32_t cs = static_cast<uint32_t>(file_size - 8); std::memcpy(p, &cs, 4); p += 4;
  std::memcpy(p, "WAVE", 4); p += 4;
  std::memcpy(p, "fmt ", 4); p += 4;
  uint32_t fs = 16; std::memcpy(p, &fs, 4); p += 4;
  uint16_t af = 1; std::memcpy(p, &af, 2); p += 2;
  uint16_t nc = 1; std::memcpy(p, &nc, 2); p += 2;
  uint32_t sr = static_cast<uint32_t>(sample_rate); std::memcpy(p, &sr, 4); p += 4;
  uint32_t br = sr * 2; std::memcpy(p, &br, 4); p += 4;
  uint16_t ba = 2; std::memcpy(p, &ba, 2); p += 2;
  uint16_t bps = 16; std::memcpy(p, &bps, 2); p += 2;
  std::memcpy(p, "data", 4); p += 4;
  uint32_t ds = static_cast<uint32_t>(data_size); std::memcpy(p, &ds, 4); p += 4;

  int16_t* dst = reinterpret_cast<int16_t*>(p);
  const float two_pi = 2.0f * 3.14159265358979323846f;
  for (int i = 0; i < n; ++i) {
    float v = 0.5f * std::sin(two_pi * freq_hz * i / sample_rate);
    dst[i] = static_cast<int16_t>(v * 32767.0f);
  }
  return wav;
}

TEST(DecodeAudioTest, BasicDecode) {
  auto wav = MakeSineWav(440.0f, 1.0f, 16000);

  const void* data_ptrs[1] = {wav.data()};
  int64_t sizes[1] = {static_cast<int64_t>(wav.size())};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  extError_t err = OrtxCreateRawAudios(raw_audios.ToBeAssigned(), data_ptrs, sizes, 1);
  ASSERT_EQ(err, kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxDecodeAudio(raw_audios.get(), 0, 0, result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  // PCM tensor
  ort_extensions::OrtxObjectPtr<OrtxTensor> pcm;
  err = OrtxTensorResultGetAt(result.get(), 0, pcm.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* pcm_data{};
  const int64_t* pcm_shape{};
  size_t pcm_dims;
  err = OrtxGetTensorData(pcm.get(), reinterpret_cast<const void**>(&pcm_data), &pcm_shape, &pcm_dims);
  ASSERT_EQ(err, kOrtxOK);

  size_t num_samples = 1;
  for (size_t d = 0; d < pcm_dims; ++d) num_samples *= pcm_shape[d];
  EXPECT_GT(num_samples, 10000ULL);  // 1s at 16kHz = 16000 samples

  // All finite, reasonable range
  for (size_t i = 0; i < num_samples; ++i) {
    ASSERT_TRUE(std::isfinite(pcm_data[i])) << "at " << i;
    ASSERT_LE(std::abs(pcm_data[i]), 2.0f) << "at " << i;
  }

  // Verify decoded PCM matches the original sine wave we generated
  const float two_pi = 2.0f * 3.14159265358979323846f;
  float max_err = 0.0f;
  for (size_t i = 0; i < num_samples; ++i) {
    float expected = 0.5f * std::sin(two_pi * 440.0f * i / 16000);
    // int16 quantization adds ~1/32768 error, allow some margin
    float err = std::abs(pcm_data[i] - expected);
    if (err > max_err) max_err = err;
  }
  // 16-bit quantization error is at most ~1/32768 ≈ 3e-5, allow 1e-3 for rounding
  EXPECT_LT(max_err, 1e-3f) << "Decoded PCM doesn't match generated sine wave";

  // Sample rate tensor
  ort_extensions::OrtxObjectPtr<OrtxTensor> sr;
  err = OrtxTensorResultGetAt(result.get(), 1, sr.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const int64_t* sr_data{};
  const int64_t* sr_shape{};
  size_t sr_dims;
  err = OrtxGetTensorData(sr.get(), reinterpret_cast<const void**>(&sr_data), &sr_shape, &sr_dims);
  ASSERT_EQ(err, kOrtxOK);
  EXPECT_EQ(*sr_data, 16000);
}

TEST(DecodeAudioTest, Resample) {
  // Create 44100 Hz WAV, decode with target 16000
  auto wav = MakeSineWav(440.0f, 0.5f, 44100);

  const void* data_ptrs[1] = {wav.data()};
  int64_t sizes[1] = {static_cast<int64_t>(wav.size())};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  ASSERT_EQ(OrtxCreateRawAudios(raw_audios.ToBeAssigned(), data_ptrs, sizes, 1), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  ASSERT_EQ(OrtxDecodeAudio(raw_audios.get(), 0, 16000, result.ToBeAssigned()), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensor> sr;
  ASSERT_EQ(OrtxTensorResultGetAt(result.get(), 1, sr.ToBeAssigned()), kOrtxOK);

  const int64_t* sr_data{};
  const int64_t* sr_shape{};
  size_t sr_dims;
  ASSERT_EQ(OrtxGetTensorData(sr.get(), reinterpret_cast<const void**>(&sr_data), &sr_shape, &sr_dims), kOrtxOK);
  EXPECT_EQ(*sr_data, 16000);

  // Check sample count is roughly 0.5s * 16000 = 8000
  ort_extensions::OrtxObjectPtr<OrtxTensor> pcm;
  ASSERT_EQ(OrtxTensorResultGetAt(result.get(), 0, pcm.ToBeAssigned()), kOrtxOK);
  const float* pcm_data{};
  const int64_t* pcm_shape{};
  size_t pcm_dims;
  ASSERT_EQ(OrtxGetTensorData(pcm.get(), reinterpret_cast<const void**>(&pcm_data), &pcm_shape, &pcm_dims), kOrtxOK);
  size_t n = 1;
  for (size_t d = 0; d < pcm_dims; ++d) n *= pcm_shape[d];
  EXPECT_NEAR(static_cast<double>(n), 8000.0, 200.0);  // allow some tolerance
}

TEST(DecodeAudioTest, InvalidIndex) {
  auto wav = MakeSineWav(440.0f, 0.1f);
  const void* data_ptrs[1] = {wav.data()};
  int64_t sizes[1] = {static_cast<int64_t>(wav.size())};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  ASSERT_EQ(OrtxCreateRawAudios(raw_audios.ToBeAssigned(), data_ptrs, sizes, 1), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  extError_t err = OrtxDecodeAudio(raw_audios.get(), 5, 0, result.ToBeAssigned());
  ASSERT_NE(err, kOrtxOK);
}

TEST(DecodeAudioTest, NullArgs) {
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  ASSERT_NE(OrtxDecodeAudio(nullptr, 0, 0, result.ToBeAssigned()), kOrtxOK);
}
