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

TEST(ExtractorTest, TestStftEnergySegmentationAndMerge) {
  const int64_t sample_rate = 16000;
  const int64_t num_samples = sample_rate * 2;

  std::vector<float> pcm(num_samples);
  const float freq = 440.0f;
  for (int64_t i = 0; i < num_samples; ++i) {
    pcm[i] = std::sin(2.0f * static_cast<float>(3.14159) * freq * (float)i / (float)sample_rate);
  }

  auto* alloc = &CppAllocator::Instance();

  ortc::Tensor<float> audio(alloc);
  float* audio_data = audio.Allocate({1, num_samples});
  std::memcpy(audio_data, pcm.data(), num_samples * sizeof(float));

  ortc::Tensor<int64_t> sr(alloc);
  sr.Allocate({1})[0] = sample_rate;

  ortc::Tensor<int64_t> frame(alloc);
  frame.Allocate({1})[0] = 25;

  ortc::Tensor<int64_t> hop(alloc);
  hop.Allocate({1})[0] = 10;

  ortc::Tensor<float> thr(alloc);
  thr.Allocate({1})[0] = -40.0f;

  ortc::Tensor<int64_t> output(alloc);

  split_signal_segments(audio, sr, frame, hop, thr, output);

  const auto& out_shape = output.Shape();
  ASSERT_EQ(out_shape.size(), 2u);
  int64_t num_segments = out_shape[0];
  ASSERT_EQ(out_shape[1], 2);
  ASSERT_EQ(num_segments, 53);

  // Start merging
  ortc::Tensor<int64_t> merge_gap(alloc);
  merge_gap.Allocate({1})[0] = 50;

  ortc::Tensor<int64_t> merged_segments(alloc);

  merge_signal_segments(output, merge_gap, merged_segments);

  const auto& merged_shape = merged_segments.Shape();
  ASSERT_EQ(merged_shape.size(), 2u);
  ASSERT_EQ(merged_shape[1], 2);
  ASSERT_EQ(merged_shape[0], 4);
}
