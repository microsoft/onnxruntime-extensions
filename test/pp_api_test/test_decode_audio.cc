// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "ortx_cpp_helper.h"
#include "ortx_extractor.h"

// Helper: generate a 16-bit PCM WAV buffer from a sine wave with N channels.
// For multi-channel output, channel `c` is given amplitude `0.5 / (c + 1)` so the
// channels are easily distinguishable in tests.
static std::vector<uint8_t> MakeSineWav(float freq_hz, float duration_sec, int sample_rate = 16000,
                                        int num_channels = 1) {
  int n = static_cast<int>(duration_sec * sample_rate);
  size_t data_size = static_cast<size_t>(n) * num_channels * 2;
  size_t file_size = 44 + data_size;
  std::vector<uint8_t> wav(file_size);
  uint8_t* p = wav.data();

  std::memcpy(p, "RIFF", 4); p += 4;
  uint32_t cs = static_cast<uint32_t>(file_size - 8);
  std::memcpy(p, &cs, 4); p += 4;
  std::memcpy(p, "WAVE", 4); p += 4;
  std::memcpy(p, "fmt ", 4); p += 4;
  uint32_t fs = 16; std::memcpy(p, &fs, 4); p += 4;
  uint16_t af = 1; std::memcpy(p, &af, 2); p += 2;
  uint16_t nc = static_cast<uint16_t>(num_channels); std::memcpy(p, &nc, 2); p += 2;
  uint32_t sr = static_cast<uint32_t>(sample_rate); std::memcpy(p, &sr, 4); p += 4;
  uint32_t br = sr * 2 * num_channels; std::memcpy(p, &br, 4); p += 4;
  uint16_t ba = static_cast<uint16_t>(2 * num_channels); std::memcpy(p, &ba, 2); p += 2;
  uint16_t bps = 16; std::memcpy(p, &bps, 2); p += 2;
  std::memcpy(p, "data", 4); p += 4;
  uint32_t ds = static_cast<uint32_t>(data_size); std::memcpy(p, &ds, 4); p += 4;

  int16_t* dst = reinterpret_cast<int16_t*>(p);
  const float two_pi = 2.0f * 3.14159265358979323846f;
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < num_channels; ++c) {
      float amp = 0.5f / static_cast<float>(c + 1);
      float v = amp * std::sin(two_pi * freq_hz * i / sample_rate);
      dst[i * num_channels + c] = static_cast<int16_t>(v * 32767.0f);
    }
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
  err = OrtxDecodeAudio(raw_audios.get(), 0, 0, /*stereo_to_mono=*/1, result.ToBeAssigned());
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
  ASSERT_EQ(OrtxDecodeAudio(raw_audios.get(), 0, 16000, /*stereo_to_mono=*/1, result.ToBeAssigned()), kOrtxOK);

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

TEST(DecodeAudioTest, NativeRatePreserved) {
  // 44100 Hz input, target_sample_rate=0 should keep native rate (no resampling)
  auto wav = MakeSineWav(440.0f, 0.5f, 44100);

  const void* data_ptrs[1] = {wav.data()};
  int64_t sizes[1] = {static_cast<int64_t>(wav.size())};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  ASSERT_EQ(OrtxCreateRawAudios(raw_audios.ToBeAssigned(), data_ptrs, sizes, 1), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  ASSERT_EQ(OrtxDecodeAudio(raw_audios.get(), 0, 0 /* native rate */, /*stereo_to_mono=*/1, result.ToBeAssigned()), kOrtxOK);

  // Sample rate should be 44100, not downsampled to 16000
  ort_extensions::OrtxObjectPtr<OrtxTensor> sr;
  ASSERT_EQ(OrtxTensorResultGetAt(result.get(), 1, sr.ToBeAssigned()), kOrtxOK);
  const int64_t* sr_data{};
  const int64_t* sr_shape{};
  size_t sr_dims;
  ASSERT_EQ(OrtxGetTensorData(sr.get(), reinterpret_cast<const void**>(&sr_data), &sr_shape, &sr_dims), kOrtxOK);
  EXPECT_EQ(*sr_data, 44100) << "target_sample_rate=0 should preserve native 44100 Hz";

  // Sample count should be ~0.5s * 44100 = 22050
  ort_extensions::OrtxObjectPtr<OrtxTensor> pcm;
  ASSERT_EQ(OrtxTensorResultGetAt(result.get(), 0, pcm.ToBeAssigned()), kOrtxOK);
  const float* pcm_data{};
  const int64_t* pcm_shape{};
  size_t pcm_dims;
  ASSERT_EQ(OrtxGetTensorData(pcm.get(), reinterpret_cast<const void**>(&pcm_data), &pcm_shape, &pcm_dims), kOrtxOK);
  size_t n = 1;
  for (size_t d = 0; d < pcm_dims; ++d) n *= pcm_shape[d];
  EXPECT_NEAR(static_cast<double>(n), 22050.0, 100.0);
}

TEST(DecodeAudioTest, InvalidIndex) {
  auto wav = MakeSineWav(440.0f, 0.1f);
  const void* data_ptrs[1] = {wav.data()};
  int64_t sizes[1] = {static_cast<int64_t>(wav.size())};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  ASSERT_EQ(OrtxCreateRawAudios(raw_audios.ToBeAssigned(), data_ptrs, sizes, 1), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  extError_t err = OrtxDecodeAudio(raw_audios.get(), 5, 0, /*stereo_to_mono=*/1, result.ToBeAssigned());
  ASSERT_NE(err, kOrtxOK);
}

TEST(DecodeAudioTest, NullArgs) {
  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  ASSERT_NE(OrtxDecodeAudio(nullptr, 0, 0, /*stereo_to_mono=*/1, result.ToBeAssigned()), kOrtxOK);
}

// Stereo input: with stereo_to_mono=1 the decoder should downmix to a single channel.
// Channel 0 is amp=0.5, channel 1 is amp=0.25; the downmix should be the average ~ 0.375 amplitude.
TEST(DecodeAudioTest, StereoDownmixToMono) {
  auto wav = MakeSineWav(440.0f, 0.25f, 16000, /*num_channels=*/2);
  const void* data_ptrs[1] = {wav.data()};
  int64_t sizes[1] = {static_cast<int64_t>(wav.size())};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  ASSERT_EQ(OrtxCreateRawAudios(raw_audios.ToBeAssigned(), data_ptrs, sizes, 1), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  ASSERT_EQ(OrtxDecodeAudio(raw_audios.get(), 0, 0, /*stereo_to_mono=*/1, result.ToBeAssigned()), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensor> pcm;
  ASSERT_EQ(OrtxTensorResultGetAt(result.get(), 0, pcm.ToBeAssigned()), kOrtxOK);
  const float* pcm_data{};
  const int64_t* pcm_shape{};
  size_t pcm_dims;
  ASSERT_EQ(OrtxGetTensorData(pcm.get(), reinterpret_cast<const void**>(&pcm_data), &pcm_shape, &pcm_dims), kOrtxOK);

  // After downmix, leading shape dimension(s) should describe a single channel.
  // The total element count should be ~0.25s * 16000 = 4000 (one channel's worth, not two).
  size_t n = 1;
  for (size_t d = 0; d < pcm_dims; ++d) n *= pcm_shape[d];
  EXPECT_NEAR(static_cast<double>(n), 4000.0, 50.0);

  // Peak amplitude should match the average of the two channels (0.5 + 0.25) / 2 = 0.375.
  float peak = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    if (std::abs(pcm_data[i]) > peak) peak = std::abs(pcm_data[i]);
  }
  EXPECT_NEAR(peak, 0.375f, 0.02f) << "Stereo downmix amplitude doesn't match expected average";
}

// Corrupt input: a buffer with a valid RIFF/WAVE header but a truncated data chunk
// (or random junk after the magic bytes) should produce a decoder error rather than a crash.
TEST(DecodeAudioTest, CorruptWav) {
  // Start from a valid WAV, then chop off most of the data chunk so the declared
  // data size is far larger than the actual buffer.
  auto wav = MakeSineWav(440.0f, 1.0f, 16000);
  ASSERT_GT(wav.size(), 60u);
  wav.resize(60);  // 44-byte header + 16 bytes of samples; declared data size still claims much more

  const void* data_ptrs[1] = {wav.data()};
  int64_t sizes[1] = {static_cast<int64_t>(wav.size())};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  ASSERT_EQ(OrtxCreateRawAudios(raw_audios.ToBeAssigned(), data_ptrs, sizes, 1), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  // Decoder may either fail or produce a short buffer; the contract we care about is
  // "doesn't crash, doesn't return garbage past the buffer". Accept both outcomes but
  // verify that on success the decoded samples don't exceed what's actually in the file.
  extError_t err = OrtxDecodeAudio(raw_audios.get(), 0, 0, /*stereo_to_mono=*/1, result.ToBeAssigned());
  if (err == kOrtxOK) {
    ort_extensions::OrtxObjectPtr<OrtxTensor> pcm;
    ASSERT_EQ(OrtxTensorResultGetAt(result.get(), 0, pcm.ToBeAssigned()), kOrtxOK);
    const float* pcm_data{};
    const int64_t* pcm_shape{};
    size_t pcm_dims;
    ASSERT_EQ(OrtxGetTensorData(pcm.get(), reinterpret_cast<const void**>(&pcm_data), &pcm_shape, &pcm_dims), kOrtxOK);
    size_t n = 1;
    for (size_t d = 0; d < pcm_dims; ++d) n *= pcm_shape[d];
    EXPECT_LE(n, 16u) << "Decoded sample count exceeded what was actually in the truncated buffer";
  }
}

// Pure garbage bytes (no valid magic for any supported format) must be rejected.
TEST(DecodeAudioTest, GarbageBytes) {
  std::vector<uint8_t> junk(1024, 0xAB);
  // Make sure no valid magic happens to appear at offset 0.
  junk[0] = 'X'; junk[1] = 'Y'; junk[2] = 'Z'; junk[3] = '0';

  const void* data_ptrs[1] = {junk.data()};
  int64_t sizes[1] = {static_cast<int64_t>(junk.size())};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  ASSERT_EQ(OrtxCreateRawAudios(raw_audios.ToBeAssigned(), data_ptrs, sizes, 1), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  ASSERT_NE(OrtxDecodeAudio(raw_audios.get(), 0, 0, /*stereo_to_mono=*/1, result.ToBeAssigned()), kOrtxOK);
}

// MP3 input via the existing test asset. Verifies that the magic-byte format detector
// in AudioDecoder works through this entry point.
TEST(DecodeAudioTest, DecodeMp3FromFile) {
  const char* paths[] = {"data/1272-141231-0002.mp3"};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  ASSERT_EQ(OrtxLoadAudios(raw_audios.ToBeAssigned(), paths, 1), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  ASSERT_EQ(OrtxDecodeAudio(raw_audios.get(), 0, 16000, /*stereo_to_mono=*/1, result.ToBeAssigned()), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensor> sr;
  ASSERT_EQ(OrtxTensorResultGetAt(result.get(), 1, sr.ToBeAssigned()), kOrtxOK);
  const int64_t* sr_data{};
  const int64_t* sr_shape{};
  size_t sr_dims;
  ASSERT_EQ(OrtxGetTensorData(sr.get(), reinterpret_cast<const void**>(&sr_data), &sr_shape, &sr_dims), kOrtxOK);
  EXPECT_EQ(*sr_data, 16000);

  ort_extensions::OrtxObjectPtr<OrtxTensor> pcm;
  ASSERT_EQ(OrtxTensorResultGetAt(result.get(), 0, pcm.ToBeAssigned()), kOrtxOK);
  const float* pcm_data{};
  const int64_t* pcm_shape{};
  size_t pcm_dims;
  ASSERT_EQ(OrtxGetTensorData(pcm.get(), reinterpret_cast<const void**>(&pcm_data), &pcm_shape, &pcm_dims), kOrtxOK);
  size_t n = 1;
  for (size_t d = 0; d < pcm_dims; ++d) n *= pcm_shape[d];
  EXPECT_GT(n, 1000u);
  for (size_t i = 0; i < n; ++i) {
    ASSERT_TRUE(std::isfinite(pcm_data[i])) << "non-finite MP3 sample at " << i;
  }
}

// FLAC input via the existing test asset.
TEST(DecodeAudioTest, DecodeFlacFromFile) {
  const char* paths[] = {"data/1272-141231-0002.flac"};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  ASSERT_EQ(OrtxLoadAudios(raw_audios.ToBeAssigned(), paths, 1), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensorResult> result;
  ASSERT_EQ(OrtxDecodeAudio(raw_audios.get(), 0, 16000, /*stereo_to_mono=*/1, result.ToBeAssigned()), kOrtxOK);

  ort_extensions::OrtxObjectPtr<OrtxTensor> sr;
  ASSERT_EQ(OrtxTensorResultGetAt(result.get(), 1, sr.ToBeAssigned()), kOrtxOK);
  const int64_t* sr_data{};
  const int64_t* sr_shape{};
  size_t sr_dims;
  ASSERT_EQ(OrtxGetTensorData(sr.get(), reinterpret_cast<const void**>(&sr_data), &sr_shape, &sr_dims), kOrtxOK);
  EXPECT_EQ(*sr_data, 16000);

  ort_extensions::OrtxObjectPtr<OrtxTensor> pcm;
  ASSERT_EQ(OrtxTensorResultGetAt(result.get(), 0, pcm.ToBeAssigned()), kOrtxOK);
  const float* pcm_data{};
  const int64_t* pcm_shape{};
  size_t pcm_dims;
  ASSERT_EQ(OrtxGetTensorData(pcm.get(), reinterpret_cast<const void**>(&pcm_data), &pcm_shape, &pcm_dims), kOrtxOK);
  size_t n = 1;
  for (size_t d = 0; d < pcm_dims; ++d) n *= pcm_shape[d];
  EXPECT_GT(n, 1000u);
  for (size_t i = 0; i < n; ++i) {
    ASSERT_TRUE(std::isfinite(pcm_data[i])) << "non-finite FLAC sample at " << i;
  }
}

// Batch decode across mixed formats: WAV (synthetic) + FLAC + MP3 in a single OrtxRawAudios.
// Verifies OrtxDecodeAudios initializes the decoder once and decodes each entry correctly.
TEST(DecodeAudioTest, BatchDecodeMixedFormats) {
  auto wav = MakeSineWav(440.0f, 0.5f, 16000);
  // Read FLAC and MP3 bytes from disk so we can pack everything into one OrtxRawAudios.
  auto load_file = [](const char* path) -> std::vector<uint8_t> {
    FILE* f = std::fopen(path, "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    size_t read = std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    buf.resize(read);
    return buf;
  };
  auto flac = load_file("data/1272-141231-0002.flac");
  auto mp3 = load_file("data/1272-141231-0002.mp3");
  ASSERT_FALSE(flac.empty());
  ASSERT_FALSE(mp3.empty());

  const void* data_ptrs[3] = {wav.data(), flac.data(), mp3.data()};
  int64_t sizes[3] = {static_cast<int64_t>(wav.size()), static_cast<int64_t>(flac.size()),
                      static_cast<int64_t>(mp3.size())};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  ASSERT_EQ(OrtxCreateRawAudios(raw_audios.ToBeAssigned(), data_ptrs, sizes, 3), kOrtxOK);

  OrtxTensorResult* results[3] = {nullptr, nullptr, nullptr};
  ASSERT_EQ(OrtxDecodeAudios(raw_audios.get(), 16000, /*stereo_to_mono=*/1, results, 3), kOrtxOK);

  for (size_t i = 0; i < 3; ++i) {
    ASSERT_NE(results[i], nullptr) << "result[" << i << "] not populated";
    ort_extensions::OrtxObjectPtr<OrtxTensorResult> holder(results[i]);  // take ownership for cleanup
    ort_extensions::OrtxObjectPtr<OrtxTensor> pcm;
    ASSERT_EQ(OrtxTensorResultGetAt(holder.get(), 0, pcm.ToBeAssigned()), kOrtxOK);
    const float* pcm_data{};
    const int64_t* pcm_shape{};
    size_t pcm_dims;
    ASSERT_EQ(OrtxGetTensorData(pcm.get(), reinterpret_cast<const void**>(&pcm_data), &pcm_shape, &pcm_dims), kOrtxOK);
    size_t n = 1;
    for (size_t d = 0; d < pcm_dims; ++d) n *= pcm_shape[d];
    EXPECT_GT(n, 100u) << "result[" << i << "] suspiciously small";

    ort_extensions::OrtxObjectPtr<OrtxTensor> sr;
    ASSERT_EQ(OrtxTensorResultGetAt(holder.get(), 1, sr.ToBeAssigned()), kOrtxOK);
    const int64_t* sr_data{};
    const int64_t* sr_shape{};
    size_t sr_dims;
    ASSERT_EQ(OrtxGetTensorData(sr.get(), reinterpret_cast<const void**>(&sr_data), &sr_shape, &sr_dims), kOrtxOK);
    EXPECT_EQ(*sr_data, 16000) << "result[" << i << "] not resampled to 16k";
  }
}

// Batch decode fail-fast: one corrupt entry should cause the whole call to fail and
// every result slot to be nullptr.
TEST(DecodeAudioTest, BatchDecodeFailFast) {
  auto good = MakeSineWav(440.0f, 0.25f, 16000);
  std::vector<uint8_t> bad(512, 0xAB);
  bad[0] = 'X'; bad[1] = 'Y'; bad[2] = 'Z'; bad[3] = '0';

  const void* data_ptrs[2] = {good.data(), bad.data()};
  int64_t sizes[2] = {static_cast<int64_t>(good.size()), static_cast<int64_t>(bad.size())};
  ort_extensions::OrtxObjectPtr<OrtxRawAudios> raw_audios;
  ASSERT_EQ(OrtxCreateRawAudios(raw_audios.ToBeAssigned(), data_ptrs, sizes, 2), kOrtxOK);

  OrtxTensorResult* results[2] = {nullptr, nullptr};
  ASSERT_NE(OrtxDecodeAudios(raw_audios.get(), 0, /*stereo_to_mono=*/1, results, 2), kOrtxOK);
  EXPECT_EQ(results[0], nullptr) << "fail-fast must clear already-produced results";
  EXPECT_EQ(results[1], nullptr);
}
