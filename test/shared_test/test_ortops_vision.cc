// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <vector>

#include "gtest/gtest.h"
// #include "opencv2/imgcodecs.hpp"
#include "vision/impl/png_encoder_decoder.hpp"

#include "ocos.h"
#include "test_kernel.hpp"

namespace {
std::vector<uint8_t> LoadBytesFromFile(const std::filesystem::path& filename) {
  using namespace std;
  ifstream ifs(filename, ios::binary | ios::ate);
  ifstream::pos_type pos = ifs.tellg();

  std::vector<uint8_t> input_bytes(pos);
  ifs.seekg(0, ios::beg);
  // we want uint8_t values so reinterpret_cast so we don't have to read chars and copy to uint8_t after.
  ifs.read(reinterpret_cast<char*>(input_bytes.data()), pos);

  return input_bytes;
}
}  // namespace

// Test DecodeImage and EncodeImage by providing a jpg image. Model will decode to BGR, encode to PNG and decode
// again to BGR. We validate that the BGR output from that matches the original image.
TEST(VisionOps, image_decode_encode) {
  std::string ort_version{OrtGetApiBase()->GetVersionString()};

  // the test model requires ONNX opset 16, which requires ORT version 1.11 or later.
  // skip test if the CI doesn't have that ORT version.
  // the CI only has a few ORT versions so we don't worry about versions <= 1.2
  if (ort_version.compare(0, 3, "1.1") != 0 ||   // earlier than 1.10
      ort_version.compare(0, 4, "1.10") == 0) {  // earlier than 1.11
    return;
  }

  auto data_dir = std::filesystem::current_path() / "data";
  auto model_path = data_dir / "ppp_vision" / "decode_encode_decode_test.onnx";
  auto image_path = data_dir / "test_colors.png";  // TEMP: Using PNG input until we have jpeg decoder implemented.

  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
  std::vector<uint8_t> image_data = LoadBytesFromFile(image_path);

  // decode image to get expected output
  const std::vector<int32_t> encoded_image_sizes{1, static_cast<int32_t>(image_data.size())};

  ort_extensions::PngDecoder decoder(image_data.data(), image_data.size());
  std::vector<uint8_t> decoded_image(decoder.NumDecodedBytes(), 0);
  assert(decoder.Decode(decoded_image.data(), decoded_image.size()));

  std::vector<TestValue> inputs{TestValue("image", image_data, {static_cast<int64_t>(image_data.size())})};
  std::vector<TestValue> outputs{TestValue("bgr_data", decoded_image, decoder.Shape())};

  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}
