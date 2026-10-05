// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <tuple>
#include <fstream>
#include <filesystem>
#include <limits>

#include "gtest/gtest.h"
#include "operators/math/energy_stft_segmentation.hpp"
#include "ortx_cpp_helper.h"
#include "shared/api/speech_extractor.h"
#include "shared/api/gemma4_audio_features.hpp"

using namespace ort_extensions;

namespace {
void CheckRawFrames(int64_t sample_count, const AttrDict& attrs, int64_t frame_size, float padding) {
  SCOPED_TRACE(sample_count);
  Gemma4Audio op;
  ASSERT_TRUE(op.Init(attrs).IsOk());
  // Deliberately use a null data pointer for the empty tensor.
  std::vector<float> samples(static_cast<size_t>(sample_count));
  for (int64_t i = 0; i < sample_count; ++i) samples[i] = static_cast<float>(i - 700) / 2048.0f;
  ortc::Tensor<float> input({1, sample_count}, sample_count ? samples.data() : nullptr);
  ortc::Tensor<float> frames(&CppAllocator::Instance());
  ortc::Tensor<bool> mask(&CppAllocator::Instance());
  ASSERT_TRUE(op.Compute(input, frames, mask).IsOk());
  const int64_t count = (sample_count + frame_size - 1) / frame_size;
  ASSERT_EQ(frames.Shape(), std::vector<int64_t>({count, frame_size}));
  ASSERT_EQ(mask.Shape(), std::vector<int64_t>({count}));
  for (int64_t i = 0; i < count * frame_size; ++i) {
    ASSERT_EQ(frames.Data()[i], i < sample_count ? samples[i] : padding) << "sample " << i;
  }
  for (int64_t i = 0; i < count; ++i) ASSERT_TRUE(mask.Data()[i]) << "frame " << i;
}
}  // namespace

TEST(ExtractorTest, TestGemma4RawControlledBoundaries) {
  for (int64_t n : {0, 1, 639, 640, 641, 1279, 1280, 1281}) {
    CheckRawFrames(n, {{"type", std::string("raw_frames")}}, 640, 0.0f);
    CheckRawFrames(n, {{"type", std::string("raw_frames")}, {"padding_value", -0.25}}, 640, -0.25f);
  }
}

TEST(ExtractorTest, TestGemma4RawFrameSizeAliases) {
  for (const auto& attrs :
       std::vector<AttrDict>{{{"type", std::string("raw_frames")}, {"feature_size", int64_t{7}}},
                             {{"type", std::string("raw_frames")}, {"audio_samples_per_token", int64_t{7}}},
                             {{"type", std::string("raw_frames")},
                              {"feature_size", int64_t{7}},
                              {"audio_samples_per_token", int64_t{7}},
                              {"sampling_rate", int64_t{8000}}}}) {
    CheckRawFrames(15, attrs, 7, 0.0f);
  }
}

TEST(ExtractorTest, TestGemma4RawRejectsUnrepresentableFrameSize) {
  float samples[] = {0.0f, 1.0f};
  for (const auto& key : {"feature_size", "audio_samples_per_token"}) {
    for (int64_t frame_size : {int64_t{4611686018427387905}, std::numeric_limits<int64_t>::max()}) {
      SCOPED_TRACE(key);
      SCOPED_TRACE(frame_size);
      Gemma4Audio op;
      ASSERT_EQ(op.Init(AttrDict{{"type", std::string("raw_frames")}, {key, frame_size}}).Code(),
                kOrtxErrorInvalidArgument);
      // Compute must also guard its state after a rejected initialization:
      // neither an overflowing byte allocation nor a huge fill may occur.
      ortc::Tensor<float> input({1, 2}, samples);
      ortc::Tensor<float> frames(&CppAllocator::Instance());
      ortc::Tensor<bool> mask(&CppAllocator::Instance());
      EXPECT_EQ(op.Compute(input, frames, mask).Code(), kOrtxErrorInvalidArgument);
      EXPECT_FALSE(static_cast<bool>(frames));
      EXPECT_FALSE(static_cast<bool>(mask));
    }
  }
}

TEST(ExtractorTest, TestGemma4RawRejectsUnrepresentableSampleCount) {
  const uint64_t max_elements =
      std::min({static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float)),
                static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max() / sizeof(float))});
  Gemma4Audio op;
  ASSERT_TRUE(op.Init(AttrDict{{"type", std::string("raw_frames")}}).IsOk());
  for (int64_t count : {int64_t{-1}, static_cast<int64_t>(max_elements + 1), std::numeric_limits<int64_t>::max()}) {
    SCOPED_TRACE(count);
    // Only the shape is inspected: this non-owning input deliberately has no
    // storage. Rejection must precede input pointer arithmetic or allocation.
    ortc::Tensor<float> input({1, count}, nullptr);
    ortc::Tensor<float> frames(&CppAllocator::Instance());
    ortc::Tensor<bool> mask(&CppAllocator::Instance());
    EXPECT_EQ(op.Compute(input, frames, mask).Code(), kOrtxErrorInvalidArgument);
    EXPECT_FALSE(static_cast<bool>(frames));
    EXPECT_FALSE(static_cast<bool>(mask));
  }

  // The input itself can fit the object limit while padding to two large
  // frames cannot. Exercise the direct raw implementation as well as dispatch.
  const int64_t frame_size = static_cast<int64_t>(max_elements / 2 + 1);
  Gemma4UnifiedAudioFrames raw;
  ASSERT_TRUE(raw.Init(AttrDict{{"feature_size", frame_size}}).IsOk());
  ortc::Tensor<float> input({1, frame_size + 1}, nullptr);
  ortc::Tensor<float> frames(&CppAllocator::Instance());
  ortc::Tensor<bool> mask(&CppAllocator::Instance());
  EXPECT_EQ(raw.Compute(input, frames, mask).Code(), kOrtxErrorInvalidArgument);
  EXPECT_FALSE(static_cast<bool>(frames));
  EXPECT_FALSE(static_cast<bool>(mask));
}

TEST(ExtractorTest, TestGemma4AudioRejectsInvalidAttributes) {
  const std::vector<AttrDict> invalid = {
      {{"type", std::string("unknown")}},
      {{"type", int64_t{1}}},
      {{"unknown_key", int64_t{1}}},
      {{"sampling_rate", int64_t{0}}},
      {{"sampling_rate", int64_t{-1}}},
      {{"sampling_rate", 16000.0}},
      {{"feature_size", std::string("128")}},
      {{"mel_floor", int64_t{1}}},
      {{"feature_size", int64_t{0}}},
      {{"feature_size", int64_t{-1}}},
      {{"per_bin_mean", std::vector<int64_t>{0}}},
      {{"type", std::string("raw_frames")}, {"feature_size", int64_t{7}}, {"audio_samples_per_token", int64_t{8}}},
      {{"type", std::string("raw_frames")}, {"feature_size", int64_t{0}}},
      {{"type", std::string("raw_frames")}, {"feature_size", int64_t{-1}}},
      {{"type", std::string("raw_frames")}, {"audio_samples_per_token", int64_t{0}}},
      {{"type", std::string("raw_frames")}, {"audio_samples_per_token", int64_t{-1}}},
      {{"type", std::string("raw_frames")}, {"audio_samples_per_token", 640.0}},
      {{"type", std::string("raw_frames")}, {"feature_size", std::vector<double>{640.0}}},
      {{"type", std::string("raw_frames")}, {"sampling_rate", int64_t{0}}},
      {{"type", std::string("raw_frames")}, {"sampling_rate", int64_t{-1}}},
      {{"type", std::string("raw_frames")}, {"sampling_rate", 16000.0}},
      {{"type", std::string("raw_frames")}, {"sampling_rate", std::string("16000")}},
      {{"type", std::string("raw_frames")}, {"padding_value", int64_t{0}}},
      {{"type", std::string("raw_frames")}, {"unknown_key", 0.0}}};
  for (size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    Gemma4Audio op;
    EXPECT_NO_THROW({
      const auto status = op.Init(invalid[i]);
      EXPECT_EQ(status.Code(), kOrtxErrorInvalidArgument) << status.Message();
    });
  }
}

TEST(ExtractorTest, TestGemma4AudioLogMelDispatcherCompatibility) {
  std::vector<float> samples(1281);
  for (size_t i = 0; i < samples.size(); ++i) samples[i] = std::sin(static_cast<float>(i) * 0.17f);
  ortc::Tensor<float> input({1, 1281}, samples.data());
  Gemma4LogMel legacy;
  ASSERT_TRUE(legacy.Init(AttrDict{}).IsOk());
  ortc::Tensor<float> expected(&CppAllocator::Instance());
  ortc::Tensor<bool> expected_mask(&CppAllocator::Instance());
  ASSERT_TRUE(legacy.Compute(input, expected, expected_mask).IsOk());
  for (const auto& attrs : std::vector<AttrDict>{{}, {{"type", std::string("log_mel")}}}) {
    Gemma4Audio op;
    ASSERT_TRUE(op.Init(attrs).IsOk());
    ortc::Tensor<float> actual(&CppAllocator::Instance());
    ortc::Tensor<bool> actual_mask(&CppAllocator::Instance());
    ASSERT_TRUE(op.Compute(input, actual, actual_mask).IsOk());
    ASSERT_EQ(actual.Shape(), expected.Shape());
    ASSERT_EQ(actual_mask.Shape(), expected_mask.Shape());
    for (int64_t i = 0; i < expected.NumberOfElement(); ++i) ASSERT_EQ(actual.Data()[i], expected.Data()[i]);
    for (int64_t i = 0; i < expected_mask.NumberOfElement(); ++i) {
      ASSERT_EQ(actual_mask.Data()[i], expected_mask.Data()[i]);
      ASSERT_TRUE(actual_mask.Data()[i]);  // Semicausal padding is left-only.
    }
  }
}

TEST(ExtractorTest, TestGemma4AudioRejectsInvalidInputShape) {
  float sample = 0.0f;
  for (const auto& mode : {"raw_frames", "log_mel"}) {
    Gemma4Audio op;
    ASSERT_TRUE(op.Init(AttrDict{{"type", std::string(mode)}}).IsOk());
    for (const auto& shape : std::vector<std::vector<int64_t>>{{}, {1}, {2, 1}, {1, 1, 1}}) {
      ortc::Tensor<float> input(shape, &sample);
      ortc::Tensor<float> features(&CppAllocator::Instance());
      ortc::Tensor<bool> mask(&CppAllocator::Instance());
      EXPECT_EQ(op.Compute(input, features, mask).Code(), kOrtxErrorInvalidArgument);
    }
  }
}

TEST(ExtractorTest, TestWhisperFeatureExtraction) {
  const char* audio_path[] = {"data/jfk.flac", "data/1272-141231-0002.wav", "data/1272-141231-0002.mp3"};
  OrtxObjectPtr<OrtxRawAudios> raw_audios;
  extError_t err = OrtxLoadAudios(raw_audios.ToBeAssigned(), audio_path, 3);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxFeatureExtractor> feature_extractor(OrtxCreateSpeechFeatureExtractor,
                                                        "data/whisper/feature_extraction.json");
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxSpeechLogMel(feature_extractor.get(), raw_audios.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 3);
  ASSERT_EQ(shape[0], 3);
  ASSERT_EQ(shape[1], 80);
  ASSERT_EQ(shape[2], 3000);
}

TEST(ExtractorTest, TestPhi4AudioFeatureExtraction) {
  const char* audio_path[] = {"data/jfk.flac", "data/1272-141231-0002.wav", "data/1272-141231-0002.mp3"};
  OrtxObjectPtr<OrtxRawAudios> raw_audios;
  extError_t err = OrtxLoadAudios(raw_audios.ToBeAssigned(), audio_path, 3);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxFeatureExtractor> feature_extractor(OrtxCreateSpeechFeatureExtractor,
                                                        "data/models/phi-4/audio_feature_extraction.json");
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxFeatureExtraction(feature_extractor.get(), raw_audios.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(std::vector<int64_t>(shape, shape + num_dims), std::vector<int64_t>({3, 1344, 80}));

  tensor.reset();
  const bool* audio_attention_mask{};
  const int64_t* audio_mask_shape{};
  size_t audio_mask_dims;
  err = OrtxTensorResultGetAt(result.get(), 1, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&audio_attention_mask), &audio_mask_shape,
                          &audio_mask_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(std::vector<int64_t>(audio_mask_shape, audio_mask_shape + audio_mask_dims),
            std::vector<int64_t>({3, 1344}));
  ASSERT_EQ(std::count(audio_attention_mask + 0 * 1344, audio_attention_mask + 1 * 1344, true), 1098);
  ASSERT_EQ(std::count(audio_attention_mask + 1 * 1344, audio_attention_mask + 2 * 1344, true), 1332);
  ASSERT_EQ(std::count(audio_attention_mask + 2 * 1344, audio_attention_mask + 3 * 1344, true), 1344);

  tensor.reset();
  err = OrtxTensorResultGetAt(result.get(), 2, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(num_dims, 1);
  ASSERT_EQ(std::vector<int64_t>(shape, shape + num_dims), std::vector<int64_t>({3}));
  const float* actual_output = reinterpret_cast<const float*>(data);
  ASSERT_FLOAT_EQ(actual_output[0], 138.0f);
  ASSERT_FLOAT_EQ(actual_output[1], 167.0f);
  ASSERT_FLOAT_EQ(actual_output[2], 168.0f);
}

TEST(ExtractorTest, TestPhi4AudioFeatureExtraction8k) {
  const char* audio_path[] = {"data/models/phi-4/1272-128104-0004-8k.wav"};
  OrtxObjectPtr<OrtxRawAudios> raw_audios;
  extError_t err = OrtxLoadAudios(raw_audios.ToBeAssigned(), audio_path, 1);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxFeatureExtractor> feature_extractor(OrtxCreateSpeechFeatureExtractor,
                                                        "data/models/phi-4/audio_feature_extraction.json");
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxFeatureExtraction(feature_extractor.get(), raw_audios.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(std::vector<int64_t>(shape, shape + num_dims), std::vector<int64_t>({1, 2938, 80}));

  tensor.reset();
  const bool* audio_attention_mask{};
  const int64_t* audio_mask_shape{};
  size_t audio_mask_dims{};
  err = OrtxTensorResultGetAt(result.get(), 1, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&audio_attention_mask), &audio_mask_shape,
                          &audio_mask_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(std::vector<int64_t>(audio_mask_shape, audio_mask_shape + audio_mask_dims),
            std::vector<int64_t>({1, 2938}));
  const size_t num_elements = std::count(audio_attention_mask, audio_attention_mask + 2938, true);
  ASSERT_EQ(num_elements, 2938);

  tensor.reset();
  err = OrtxTensorResultGetAt(result.get(), 2, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(num_dims, 1);
  ASSERT_EQ(std::vector<int64_t>(shape, shape + num_dims), std::vector<int64_t>({1}));
}

TEST(ExtractorTest, TestPhi4AudioOutput) {
  const char* audio_path[] = {"data/1272-141231-0002.wav"};
  OrtxObjectPtr<OrtxRawAudios> raw_audios;
  extError_t err = OrtxLoadAudios(raw_audios.ToBeAssigned(), audio_path, 1);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxFeatureExtractor> feature_extractor(OrtxCreateSpeechFeatureExtractor,
                                                        "data/models/phi-4/audio_feature_extraction.json");
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxFeatureExtraction(feature_extractor.get(), raw_audios.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(std::vector<int64_t>(shape, shape + num_dims), std::vector<int64_t>({1, 1332, 80}));

  // Dimensions
  const size_t num_rows = shape[1];
  const size_t num_columns = shape[2];

  // Read the expected output from the file
  std::filesystem::path expected_audio_embed_output_path = "data/models/phi-4/expected_output.txt";
  std::ifstream expected_audio_embed_output(expected_audio_embed_output_path);

  ASSERT_TRUE(expected_audio_embed_output.is_open());

  // Define lambda for comparison
  auto are_close = [](float a, float b, float rtol = 1e-03, float atol = 1e-02) -> bool {
    return std::abs(a - b) <= atol || std::abs(a - b) <= rtol * std::abs(b);
  };

  size_t num_mismatched = 0;
  size_t total_elements = num_rows * 10;  // We only compare the first 10 columns
  std::string line;
  size_t row_idx = 0;

  while (std::getline(expected_audio_embed_output, line) && row_idx < num_rows) {
    std::stringstream ss(line);  // Stringstream to parse each line
    std::string value_str;
    size_t col_idx = 0;

    while (std::getline(ss, value_str, ',') && col_idx < 10) {  // Only read the first 10 columns
      float expected_value = std::stof(value_str);              // Convert string to float

      // Compare values
      const float* row_start = data + (row_idx * num_columns);
      if (!are_close(row_start[col_idx], expected_value)) {
        num_mismatched++;  // Count mismatches
        std::cout << "Mismatch at (" << row_idx << "," << col_idx << "): "
                  << "Expected: " << expected_value << ", Got: " << row_start[col_idx] << std::endl;
      }
      col_idx++;
    }
    row_idx++;
  }

  expected_audio_embed_output.close();

  // Calculate the mismatch percentage
  float mismatch_percentage = static_cast<float>(num_mismatched) / total_elements;

  std::cout << "Mismatch percentage: " << mismatch_percentage * 100 << "%" << std::endl;

  // We use a 2% mismatch threshold, same as that for Whisper
  ASSERT_LT(mismatch_percentage, 0.02) << "Mismatch percentage exceeds 2% threshold!";
}

TEST(ExtractorTest, TestWhisperAudioOutput) {
  const char* audio_path[] = {"data/1272-141231-0002.flac"};
  OrtxObjectPtr<OrtxRawAudios> raw_audios;
  extError_t err = OrtxLoadAudios(raw_audios.ToBeAssigned(), audio_path, 1);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxFeatureExtractor> feature_extractor(OrtxCreateSpeechFeatureExtractor,
                                                        "data/whisper/feature_extraction.json");
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxFeatureExtraction(feature_extractor.get(), raw_audios.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(std::vector<int64_t>(shape, shape + num_dims), std::vector<int64_t>({1, 80, 3000}));

  // Dimensions
  const size_t num_rows = shape[1];
  const size_t num_columns = shape[2];

  // Read the expected output from the file
  std::filesystem::path expected_audio_embed_output_path = "data/whisper/expected_output.txt";
  std::ifstream expected_audio_embed_output(expected_audio_embed_output_path);

  ASSERT_TRUE(expected_audio_embed_output.is_open());

  // Define lambda for comparison
  auto are_close = [](float a, float b, float rtol = 1e-03, float atol = 1e-02) -> bool {
    return std::abs(a - b) <= atol || std::abs(a - b) <= rtol * std::abs(b);
  };

  size_t num_mismatched = 0;
  size_t total_elements = num_rows * 10;  // We only compare the first 10 columns
  std::string line;
  size_t row_idx = 0;

  while (std::getline(expected_audio_embed_output, line) && row_idx < num_rows) {
    std::stringstream ss(line);  // Stringstream to parse each line
    std::string value_str;
    size_t col_idx = 0;

    while (std::getline(ss, value_str, ',') && col_idx < 10) {  // Only read the first 10 columns
      float expected_value = std::stof(value_str);              // Convert string to float

      // Compare values
      const float* row_start = data + (row_idx * num_columns);
      if (!are_close(row_start[col_idx], expected_value)) {
        num_mismatched++;  // Count mismatches
        std::cout << "Mismatch at (" << row_idx << "," << col_idx << "): "
                  << "Expected: " << expected_value << ", Got: " << row_start[col_idx] << std::endl;
      }
      col_idx++;
    }
    row_idx++;
  }

  expected_audio_embed_output.close();

  // Calculate the mismatch percentage
  float mismatch_percentage = static_cast<float>(num_mismatched) / total_elements;

  std::cout << "Mismatch percentage: " << mismatch_percentage * 100 << "%" << std::endl;

  // We use a 4% mismatch threshold currently, and aim to improve this further in the future
  ASSERT_LT(mismatch_percentage, 0.04) << "Mismatch percentage exceeds 4% threshold!";
}

TEST(ExtractorTest, TestSplitSignalSegments) {
  const int64_t sample_rate = 16000;
  const int64_t num_samples = sample_rate * 2;

  std::vector<float> pcm(num_samples);
  const float freq = 440.0f;
  for (int64_t i = 0; i < num_samples; ++i) {
    pcm[i] = std::sin(2.0f * static_cast<float>(3.14159) * freq * (float)i / (float)sample_rate);
  }

  auto* alloc = &CppAllocator::Instance();

  ortc::Tensor<float> input(alloc);
  float* in_data = input.Allocate({1, num_samples});
  std::memcpy(in_data, pcm.data(), num_samples * sizeof(float));

  ortc::Tensor<int64_t> sr(alloc);
  sr.Allocate({1})[0] = sample_rate;

  ortc::Tensor<int64_t> frame_ms(alloc);
  frame_ms.Allocate({1})[0] = 25;

  ortc::Tensor<int64_t> hop_ms(alloc);
  hop_ms.Allocate({1})[0] = 10;

  ortc::Tensor<float> energy_threshold_db(alloc);
  // Difference of 40 decibels can be a reasonable diff between voice and silence (or background noise)
  energy_threshold_db.Allocate({1})[0] = -40.0f;

  ortc::Tensor<int64_t> output(alloc);

  extError_t err = OrtxSplitSignalSegments(
      reinterpret_cast<OrtxTensor*>(&input), reinterpret_cast<OrtxTensor*>(&sr),
      reinterpret_cast<OrtxTensor*>(&frame_ms), reinterpret_cast<OrtxTensor*>(&hop_ms),
      reinterpret_cast<OrtxTensor*>(&energy_threshold_db), reinterpret_cast<OrtxTensor*>(&output));

  ASSERT_EQ(err, kOrtxOK);

  const auto& out_shape = output.Shape();
  ASSERT_EQ(out_shape.size(), 2u);
  ASSERT_EQ(out_shape[1], 2);
  ASSERT_EQ(out_shape[0], 53);

  ortc::Tensor<int64_t> merge_gap(alloc);
  merge_gap.Allocate({1})[0] = 50;

  ortc::Tensor<int64_t> merged_segments(alloc);

  err = OrtxMergeSignalSegments(reinterpret_cast<OrtxTensor*>(&output), reinterpret_cast<OrtxTensor*>(&merge_gap),
                                reinterpret_cast<OrtxTensor*>(&merged_segments));

  ASSERT_EQ(err, kOrtxOK);

  const auto& merged_shape = merged_segments.Shape();
  ASSERT_EQ(merged_shape.size(), 2u);
  ASSERT_EQ(merged_shape[1], 2);
  ASSERT_EQ(merged_shape[0], 4);
}

TEST(ExtractorTest, TestGemma4AudioFeatureExtraction) {
  // Use existing test audio files to verify the Gemma 4 USM-style log-mel pipeline:
  // AudioDecoder -> Gemma4Audio (type="log_mel")
  const char* audio_path[] = {"data/jfk.flac"};
  OrtxObjectPtr<OrtxRawAudios> raw_audios;
  extError_t err = OrtxLoadAudios(raw_audios.ToBeAssigned(), audio_path, 1);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxFeatureExtractor> feature_extractor(OrtxCreateSpeechFeatureExtractor,
                                                        "data/models/gemma-4/audio_feature_extraction.json");
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxFeatureExtraction(feature_extractor.get(), raw_audios.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  // Output 0: log-mel spectrogram — float (batch, num_frames, 128)
  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 3ULL);           // (batch, num_frames, feature_size)
  ASSERT_EQ(shape[0], 1);              // single audio
  ASSERT_EQ(shape[2], 128);            // 128 mel bins
  EXPECT_GT(shape[1], 0);              // should have some frames

  // Verify values are finite (not NaN/Inf).
  for (int64_t i = 0; i < std::min<int64_t>(shape[1] * 128, 5000); ++i) {
    ASSERT_TRUE(std::isfinite(data[i])) << "log-mel value at index " << i << " is not finite";
  }

  // Verify log-mel values are in a reasonable range.
  // With mel_floor=0.001, log(0.001) ~ -6.9078. Values should be >= ~-7
  // and typically < ~5 for speech audio.
  const float* frame0 = data;
  for (int i = 0; i < 10; ++i) {
    EXPECT_GE(frame0[i], -7.5f) << "Frame 0 bin " << i << " too low";
    EXPECT_LE(frame0[i], 5.0f) << "Frame 0 bin " << i << " too high";
  }
  // The first mel bin of silent/padding frames should be close to log(mel_floor)
  // = log(0.001) ~ -6.9078.
  EXPECT_NEAR(frame0[0], -6.9078f, 0.05f)
      << "Frame 0 bin 0 should be near log(0.001) for semicausal pad region";

  // Output 1: attention mask — bool (batch, num_frames)
  err = OrtxTensorResultGetAt(result.get(), 1, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const bool* mask_data{};
  const int64_t* mask_shape{};
  size_t mask_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&mask_data), &mask_shape, &mask_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(mask_dims, 2ULL);          // (batch, num_frames)
  ASSERT_EQ(mask_shape[0], 1);
  ASSERT_EQ(mask_shape[1], shape[1]);   // same frame count

  // For JFK audio (not truncated), all frames except those from the semicausal pad
  // should be valid. At least some frames should be true.
  int true_count = std::count(mask_data, mask_data + mask_shape[1], true);
  EXPECT_GT(true_count, 0) << "Expected at least some valid frames";
}

TEST(ExtractorTest, TestGemma4AudioFeatureExtractionMultiFile) {
  // Verify batched processing with multiple audio files.
  const char* audio_path[] = {"data/jfk.flac", "data/1272-141231-0002.wav"};
  OrtxObjectPtr<OrtxRawAudios> raw_audios;
  extError_t err = OrtxLoadAudios(raw_audios.ToBeAssigned(), audio_path, 2);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxFeatureExtractor> feature_extractor(OrtxCreateSpeechFeatureExtractor,
                                                        "data/models/gemma-4/audio_feature_extraction.json");
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxFeatureExtraction(feature_extractor.get(), raw_audios.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  // log-mel: batch dim should be 2
  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 3ULL);
  ASSERT_EQ(shape[0], 2);             // batch of 2
  ASSERT_EQ(shape[2], 128);

  // mask: batch dim should be 2
  err = OrtxTensorResultGetAt(result.get(), 1, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const bool* mask_data{};
  const int64_t* mask_shape{};
  size_t mask_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&mask_data), &mask_shape, &mask_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(mask_dims, 2ULL);
  ASSERT_EQ(mask_shape[0], 2);
}

// Regression test: a Gemma4LogMel config whose per_bin_mean / per_bin_stddev
// length does not match feature_size must be rejected at Init time. Previously
// Compute() indexed these vectors by m in [0, feature_size_) guarded only by
// !empty(), causing an out-of-bounds heap read (and heap disclosure through the
// output tensor) for every audio frame.
TEST(ExtractorTest, TestGemma4LogMelRejectsMismatchedNormalizationLength) {
  // feature_size = 128 but per_bin_mean has a single element.
  const char* mismatched_mean_def = R"({
    "feature_extraction": {
      "sequence": [
        {
          "operation": {
            "name": "gemma4_log_mel",
            "type": "Gemma4LogMel",
            "attrs": {
              "feature_size": 128,
              "sampling_rate": 16000,
              "frame_length_ms": 20.0,
              "hop_length_ms": 10.0,
              "per_bin_mean": [0.0],
              "per_bin_stddev": [1.0]
            }
          }
        }
      ]
    }
  })";

  OrtxFeatureExtractor* extractor = nullptr;
  extError_t err = OrtxCreateSpeechFeatureExtractor(&extractor, mismatched_mean_def);
  EXPECT_NE(err, kOrtxOK) << "Init should reject per_bin_mean length != feature_size";
  EXPECT_EQ(extractor, nullptr);
  if (extractor != nullptr) {
    OrtxDisposeOnly(extractor);
  }

  // feature_size = 128, per_bin_mean matches but per_bin_stddev is too short.
  const char* mismatched_stddev_def = R"({
    "feature_extraction": {
      "sequence": [
        {
          "operation": {
            "name": "gemma4_log_mel",
            "type": "Gemma4LogMel",
            "attrs": {
              "feature_size": 3,
              "sampling_rate": 16000,
              "frame_length_ms": 20.0,
              "hop_length_ms": 10.0,
              "per_bin_mean": [0.0, 0.0, 0.0],
              "per_bin_stddev": [1.0]
            }
          }
        }
      ]
    }
  })";

  extractor = nullptr;
  err = OrtxCreateSpeechFeatureExtractor(&extractor, mismatched_stddev_def);
  EXPECT_NE(err, kOrtxOK) << "Init should reject per_bin_stddev length != feature_size";
  EXPECT_EQ(extractor, nullptr);
  if (extractor != nullptr) {
    OrtxDisposeOnly(extractor);
  }

  // Sanity check: a correctly-sized config is accepted.
  const char* matched_def = R"({
    "feature_extraction": {
      "sequence": [
        {
          "operation": {
            "name": "gemma4_log_mel",
            "type": "Gemma4LogMel",
            "attrs": {
              "feature_size": 3,
              "sampling_rate": 16000,
              "frame_length_ms": 20.0,
              "hop_length_ms": 10.0,
              "per_bin_mean": [0.0, 0.0, 0.0],
              "per_bin_stddev": [1.0, 1.0, 1.0]
            }
          }
        }
      ]
    }
  })";

  extractor = nullptr;
  err = OrtxCreateSpeechFeatureExtractor(&extractor, matched_def);
  EXPECT_EQ(err, kOrtxOK) << "Init should accept matching normalization lengths";
  EXPECT_NE(extractor, nullptr);
  if (extractor != nullptr) {
    OrtxDisposeOnly(extractor);
  }
}

TEST(ExtractorTest, TestGemma4UnifiedAudioFrames) {
  // gemma-4-12B "unified" (encoder-free) audio: raw 16 kHz waveform chunked
  // into fixed 640-sample frames via the generic Gemma4Audio op with
  // type="raw_frames".  Pipeline: AudioDecoder -> Gemma4Audio
  const char* audio_path[] = {"data/jfk.flac"};
  OrtxObjectPtr<OrtxRawAudios> raw_audios;
  extError_t err = OrtxLoadAudios(raw_audios.ToBeAssigned(), audio_path, 1);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxFeatureExtractor> feature_extractor(
      OrtxCreateSpeechFeatureExtractor, "data/models/gemma-4-unified/audio_feature_extraction.json");
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxFeatureExtraction(feature_extractor.get(), raw_audios.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  // Output 0: raw waveform frames — float (batch, num_tokens, 640)
  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 3ULL);  // (batch, num_tokens, samples_per_token)
  ASSERT_EQ(shape[0], 1);     // single audio
  ASSERT_EQ(shape[2], 640);   // 640 raw samples per token
  EXPECT_GT(shape[1], 0);     // at least one frame
  const int64_t num_tokens = shape[1];

  // Raw waveform frames are the decoded PCM samples, which the AudioDecoder
  // normalizes to [-1, 1]; a small epsilon covers float rounding at full scale.
  for (int64_t i = 0; i < std::min<int64_t>(num_tokens * 640, 5000); ++i) {
    ASSERT_TRUE(std::isfinite(data[i])) << "frame value at index " << i << " is not finite";
    ASSERT_LE(std::abs(data[i]), 1.0001f) << "frame value at index " << i << " out of normalized PCM range";
  }

  // Output 1: frame mask — bool (batch, num_tokens), all true for a single clip.
  err = OrtxTensorResultGetAt(result.get(), 1, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  const bool* mask_data{};
  const int64_t* mask_shape{};
  size_t mask_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&mask_data), &mask_shape, &mask_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(mask_dims, 2ULL);            // (batch, num_tokens)
  ASSERT_EQ(mask_shape[0], 1);
  ASSERT_EQ(mask_shape[1], num_tokens);  // same frame count as features
  for (int64_t i = 0; i < num_tokens; ++i) {
    EXPECT_TRUE(mask_data[i]) << "single-clip frame " << i << " should be valid";
  }
}

TEST(ExtractorTest, TestGemma4UnifiedAudioFramesMultiFile) {
  // Two clips of different lengths: verify batch stacking pads the shorter clip's
  // frames and that the frame mask marks the padded tail invalid (false), while
  // the real frames of each clip are valid (true). Locks in the unified batch +
  // mask behavior, mirroring the log-mel multi-file coverage.
  const char* audio_path[] = {"data/jfk.flac", "data/1272-141231-0002.wav"};
  OrtxObjectPtr<OrtxRawAudios> raw_audios;
  extError_t err = OrtxLoadAudios(raw_audios.ToBeAssigned(), audio_path, 2);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxFeatureExtractor> feature_extractor(
      OrtxCreateSpeechFeatureExtractor, "data/models/gemma-4-unified/audio_feature_extraction.json");
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxFeatureExtraction(feature_extractor.get(), raw_audios.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  // Output 0: frames — float (2, max_tokens, 640)
  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 3ULL);
  ASSERT_EQ(shape[0], 2);    // batch of 2 clips
  ASSERT_EQ(shape[2], 640);  // raw samples per token
  const int64_t max_tokens = shape[1];

  // Output 1: mask — bool (2, max_tokens)
  err = OrtxTensorResultGetAt(result.get(), 1, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  const bool* mask_data{};
  const int64_t* mask_shape{};
  size_t mask_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&mask_data), &mask_shape, &mask_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(mask_dims, 2ULL);
  ASSERT_EQ(mask_shape[0], 2);
  ASSERT_EQ(mask_shape[1], max_tokens);

  // Each row's mask must be a contiguous true-prefix (real frames) followed by a
  // false-suffix (batch padding). Count valid frames per clip.
  int64_t valid_counts[2] = {0, 0};
  for (int64_t b = 0; b < 2; ++b) {
    const bool* row = mask_data + b * max_tokens;
    bool seen_false = false;
    for (int64_t i = 0; i < max_tokens; ++i) {
      if (row[i]) {
        ASSERT_FALSE(seen_false) << "clip " << b << " mask must not have a true frame after padding";
        ++valid_counts[b];
      } else {
        seen_false = true;
      }
    }
    EXPECT_GT(valid_counts[b], 0) << "clip " << b << " should have at least one valid frame";
  }

  // The two clips have different lengths, so exactly one clip fills all max_tokens
  // and the shorter clip has a padded (false) tail.
  EXPECT_NE(valid_counts[0], valid_counts[1]) << "test clips should differ in length";
  EXPECT_EQ(std::max(valid_counts[0], valid_counts[1]), max_tokens);
  const int64_t shorter = std::min(valid_counts[0], valid_counts[1]);
  EXPECT_LT(shorter, max_tokens) << "shorter clip should be zero-padded in the batch";

  // Separate extraction supplies exact frame counts and every real sample,
  // rather than inferring correctness solely from the batch's own mask.
  for (int64_t b = 0; b < 2; ++b) {
    OrtxObjectPtr<OrtxRawAudios> single_audio;
    ASSERT_EQ(OrtxLoadAudios(single_audio.ToBeAssigned(), audio_path + b, 1), kOrtxOK);
    OrtxObjectPtr<OrtxTensorResult> single_result;
    ASSERT_EQ(OrtxFeatureExtraction(feature_extractor.get(), single_audio.get(), single_result.ToBeAssigned()),
              kOrtxOK);
    OrtxObjectPtr<OrtxTensor> single_tensor;
    ASSERT_EQ(OrtxTensorResultGetAt(single_result.get(), 0, single_tensor.ToBeAssigned()), kOrtxOK);
    const float* single_data{};
    const int64_t* single_shape{};
    size_t single_dims{};
    ASSERT_EQ(OrtxGetTensorData(single_tensor.get(), reinterpret_cast<const void**>(&single_data), &single_shape,
                                &single_dims),
              kOrtxOK);
    ASSERT_EQ(single_dims, 3ULL);
    ASSERT_EQ(single_shape[0], 1);
    ASSERT_EQ(single_shape[2], 640);
    const int64_t single_count = single_shape[1];
    ASSERT_EQ(valid_counts[b], single_count);
    for (int64_t t = 0; t < max_tokens; ++t) {
      ASSERT_EQ(mask_data[b * max_tokens + t], t < single_count);
      for (int64_t s = 0; s < 640; ++s) {
        ASSERT_EQ(data[(b * max_tokens + t) * 640 + s], t < single_count ? single_data[t * 640 + s] : 0.0f);
      }
    }
  }
}
