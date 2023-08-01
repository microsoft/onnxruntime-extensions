// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_AZURE
#include <cstdlib>

#include "gtest/gtest.h"

#include "ocos.h"
#include "narrow.h"
#include "test_kernel.hpp"
#include "test_utils.hpp"

using namespace ort_extensions;
using namespace ort_extensions::test;

// Test custom op with OpenAIAudioInvoker calling Whisper
TEST(AzureOps, OpenAIWhisper) {
  std::string ort_version{OrtGetApiBase()->GetVersionString()};

  auto data_dir = std::filesystem::current_path() / "data" / "AzureEP";
  auto model_path = data_dir / "openai_whisper_transcriptions.onnx";
  auto audio_path = data_dir / "self-destruct-button.wav";
  std::vector<uint8_t> audio_data = LoadBytesFromFile(audio_path);

  const char* auth_token = std::getenv("OPENAI_AUTH_TOKEN");
  if (auth_token == nullptr) {
    GTEST_SKIP() << "OPENAI_AUTH_TOKEN environment variable was not set.";
  }

  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs{TestValue("auth_token", {std::string(auth_token)}, {1}),
                                TestValue("transcribe0/file", audio_data, {narrow<int64_t>(audio_data.size())})};
  // punctuation can differ between calls to OpenAI Whisper (sometimes there's a comma after 'button' and sometimes
  // a full stop) so we use a custom output validator that looks for substrings in the output.
  std::vector<std::string> expected_output{"Thank you for pressing the self-destruct button",
                                           "ship will self-destruct in three minutes"};

  OutputValidator find_strings_in_output = [&expected_output](size_t output_idx, Ort::Value& actual, TestValue expected) {
    std::vector<std::string> output_string;
    GetTensorMutableDataString(Ort::GetApi(), actual, output_string);

    ASSERT_EQ(output_string.size(), 1) << "Expected the Whisper response to be a single string with json";

    for (auto& expected_substring : expected_output) {
      EXPECT_NE(output_string[0].find(expected_substring), std::string::npos)
          << "'" << expected_substring << "' was not found in output " << output_string[0];
    }
  };

  std::vector<TestValue> outputs{TestValue("transcriptions", expected_output, {1})};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath(), find_strings_in_output);

  // Use mp3 input (despite the node having audio_format set to 'wav') and test altering the spelling using a prompt
  audio_path = data_dir / "be-a-man-take-some-pepto-bismol-get-dressed-and-come-on-over-here.mp3";
  audio_data = LoadBytesFromFile(audio_path);
  inputs[1] = TestValue("transcribe0/file", audio_data, {narrow<int64_t>(audio_data.size())});
  expected_output = {"Take some Pepto-Bismol, get dressed, and come on over here."};

  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath(), find_strings_in_output);

  // use prompt to mis-spell Pepto-Bismol
  std::string prompt = "Peptoe-Bismole.";
  inputs.push_back(TestValue("transcribe0/prompt", {prompt}, {1}));
  expected_output = {"Take some Peptoe-Bismole, get dressed, and come on over here."};
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath(), find_strings_in_output);
}

#endif
