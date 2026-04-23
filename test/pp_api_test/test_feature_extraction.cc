// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <tuple>
#include <fstream>
#include <filesystem>

#include "gtest/gtest.h"
#include "operators/math/energy_stft_segmentation.hpp"
#include "ortx_cpp_helper.h"
#include "shared/api/speech_extractor.h"

using namespace ort_extensions;

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
  // AudioDecoder -> Gemma4LogMel
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