// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_CV2

#include <filesystem>
#include <fstream>
#include <vector>

#include "gtest/gtest.h"
#include "opencv2/imgcodecs.hpp"

#include "ocos.h"
#include "test_kernel.hpp"
#include "utils.hpp"

using namespace ort_extensions::test;

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
  auto image_path = data_dir / "test_colors.jpg";

  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
  std::vector<uint8_t> image_data = LoadBytesFromFile(image_path);

  // decode image to get expected output
  const std::vector<int32_t> encoded_image_sizes{1, static_cast<int32_t>(image_data.size())};
  const cv::Mat encoded_image(encoded_image_sizes, CV_8UC1, static_cast<void*>(image_data.data()));
  const cv::Mat decoded_image = cv::imdecode(encoded_image, cv::IMREAD_COLOR);
  ASSERT_NE(decoded_image.data, nullptr) << "imdecode failed";

  const cv::Size decoded_image_size = decoded_image.size();
  const int64_t colors = 3;
  const std::vector<int64_t> output_dimensions{decoded_image_size.height, decoded_image_size.width, colors};
  // decoded_image.total() is num pixels. elemSize is 3 (BGR value per pixel)
  const auto num_output_bytes = decoded_image.total() * decoded_image.elemSize();
  std::vector<uint8_t> expected_output(num_output_bytes, 0);
  memcpy(expected_output.data(), decoded_image.data, num_output_bytes);

  std::vector<TestValue> inputs{TestValue("image", image_data, {static_cast<int64_t>(image_data.size())})};
  std::vector<TestValue> outputs{TestValue("bgr_data", expected_output, output_dimensions)};

  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

#endif
