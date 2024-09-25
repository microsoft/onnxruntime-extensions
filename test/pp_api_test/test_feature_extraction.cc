// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <tuple>
#include <fstream>
#include <filesystem>

#include "gtest/gtest.h"
#include "ortx_cpp_helper.h"
#include "shared/api/speech_extractor.h"

using namespace ort_extensions;

TEST(ExtractorTest, TestWhisperFeatureExtraction) {
  const char* audio_path[] = {"data/jfk.flac", "data/1272-141231-0002.wav", "data/1272-141231-0002.mp3"};
  OrtxObjectPtr<OrtxRawAudios> raw_audios;
  extError_t err = OrtxLoadAudios(ort_extensions::ptr(raw_audios), audio_path, 3);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxFeatureExtractor> feature_extractor(OrtxCreateSpeechFeatureExtractor, "data/whisper/feature_extraction.json");
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxSpeechLogMel(feature_extractor.get(), raw_audios.get(), ort_extensions::ptr(result));
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, ort_extensions::ptr(tensor));
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
