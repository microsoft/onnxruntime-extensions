// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_AZURE
#include <cstdlib>

#include "gtest/gtest.h"

#include "ocos.h"
#include "narrow.h"
#include "test_kernel.hpp"
#include "utils.hpp"

using namespace ort_extensions;
using namespace ort_extensions::test;

// Test custom op with OpenAIAudioInvoker calling Whisper
// Default input format. No prompt.
TEST(AzureOps, OpenAIWhisper_basic) {
  const char* auth_token = std::getenv("OPENAI_AUTH_TOKEN");
  if (auth_token == nullptr) {
    GTEST_SKIP() << "OPENAI_AUTH_TOKEN environment variable was not set.";
  }

  auto data_dir = std::filesystem::current_path() / "data" / "azure";
  auto model_path = data_dir / "openai_whisper_transcriptions.onnx";
  auto audio_path = data_dir / "self-destruct-button.wav";
  std::vector<uint8_t> audio_data = LoadBytesFromFile(audio_path);

  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs{TestValue("auth_token", {std::string(auth_token)}, {1}),
                                TestValue("transcribe0/file", audio_data, {narrow<int64_t>(audio_data.size())})};

  // punctuation can differ between calls to OpenAI Whisper. sometimes there's a comma after 'button' and sometimes
  // a full stop. use a custom output validator that looks for substrings in the output that aren't affected by this.
  std::vector<std::string> expected_output{"Thank you for pressing the self-destruct button",
                                           "ship will self-destruct in three minutes"};

  // dims are set to '{1}' as we expect one string output. the expected_output is the collection of substrings to look
  // for in the single output
  std::vector<TestValue> outputs{TestValue("transcription", expected_output, {1})};

  OutputValidator find_strings_in_output =
      [](size_t output_idx, Ort::Value& actual, TestValue expected) {
        std::vector<std::string> output_string;
        GetTensorMutableDataString(Ort::GetApi(), actual, output_string);

        ASSERT_EQ(output_string.size(), 1) << "Expected the Whisper response to be a single string with json";

        for (auto& expected_substring : expected.values_string) {
          if (output_string[0].find(expected_substring) == std::string::npos) {
            FAIL() << "'" << expected_substring << "' was not found in output " << output_string[0];
          }
        }
      };

  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath(), find_strings_in_output);
}

// test calling Whisper with a filename to provide mp3 instead of the default wav, and the optional prompt
TEST(AzureOps, OpenAIWhisper_Prompt_CustomFormat) {
  const char* auth_token = std::getenv("OPENAI_AUTH_TOKEN");
  if (auth_token == nullptr) {
    GTEST_SKIP() << "OPENAI_AUTH_TOKEN environment variable was not set.";
  }

  std::string ort_version{OrtGetApiBase()->GetVersionString()};

  auto data_dir = std::filesystem::current_path() / "data" / "azure";
  auto model_path = data_dir / "openai_whisper_transcriptions.onnx";
  auto audio_path = data_dir / "be-a-man-take-some-pepto-bismol-get-dressed-and-come-on-over-here.mp3";
  std::vector<uint8_t> audio_data = LoadBytesFromFile(audio_path);

  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  // provide filename with 'mp3' extension to indicate audio format. doesn't need to be the 'real' filename
  std::vector<TestValue> inputs{TestValue("auth_token", {std::string(auth_token)}, {1}),
                                TestValue("transcribe0/file", audio_data, {narrow<int64_t>(audio_data.size())}),
                                TestValue("transcribe0/filename", {std::string("audio.mp3")}, {1})};

  std::vector<std::string> expected_output = {"Take some Pepto-Bismol, get dressed, and come on over here."};
  std::vector<TestValue> outputs{TestValue("transcription", expected_output, {1})};

  OutputValidator find_strings_in_output =
      [](size_t output_idx, Ort::Value& actual, TestValue expected) {
        std::vector<std::string> output_string;
        GetTensorMutableDataString(Ort::GetApi(), actual, output_string);

        ASSERT_EQ(output_string.size(), 1) << "Expected the Whisper response to be a single string with json";
        const auto& expected_substring = expected.values_string[0];
        if (output_string[0].find(expected_substring) == std::string::npos) {
          FAIL() << "'" << expected_substring << "' was not found in output " << output_string[0];
        }
      };

  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath(), find_strings_in_output);

  // use optional 'prompt' input to mis-spell Pepto-Bismol in response
  std::string prompt = "Peptoe-Bismole";
  inputs.push_back(TestValue("transcribe0/prompt", {prompt}, {1}));
  outputs[0].values_string[0] = "Take some Peptoe-Bismole, get dressed, and come on over here.";
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath(), find_strings_in_output);
}

#endif
