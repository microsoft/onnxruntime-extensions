// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <tuple>
#include <fstream>
#include <filesystem>

#include "gtest/gtest.h"
#include "shared/api/c_api_utils.hpp"

#include "vision/decode_image.hpp"


using namespace ort_extensions;

TEST(ImgDecoderTest, TestPngDecoder) {
  ort_extensions::DecodeImage image_decoder;
  image_decoder.Init(std::unordered_map<std::string, std::variant<std::string>>());
  std::vector<uint8_t> png_data;
  std::filesystem::path png_path = "data/processor/exceltable.png";
  std::ifstream png_file(png_path, std::ios::binary);
  ASSERT_TRUE(png_file.is_open());
  png_file.seekg(0, std::ios::end);
  png_data.resize(png_file.tellg());
  png_file.seekg(0, std::ios::beg);
  png_file.read(reinterpret_cast<char*>(png_data.data()), png_data.size());
  png_file.close();

  ortc::Tensor<uint8_t> png_tensor({static_cast<int64_t>(png_data.size())},  png_data.data());
  ortc::Tensor<uint8_t> out_tensor{&CppAllocator::Instance()};
  auto status = image_decoder.Compute(png_tensor, out_tensor);
  ASSERT_TRUE(status.IsOk()) << status.ToString();

  ASSERT_EQ(out_tensor.Shape(), std::vector<int64_t>({206, 487, 3}));
  auto out_range = out_tensor.Data() + 0;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

  out_range = out_tensor.Data() + 477 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

  out_range = out_tensor.Data() + 243 * 206 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217}));

  out_range = out_tensor.Data() + 485 * 206 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ImageDecoderTest, TestJpegDecoder) {
  ort_extensions::DecodeImage image_decoder;
  image_decoder.Init(std::unordered_map<std::string, std::variant<std::string>>());
  std::vector<uint8_t> jpeg_data;
  std::filesystem::path jpeg_path = "data/processor/australia.jpg";
  std::ifstream jpeg_file(jpeg_path, std::ios::binary);
  ASSERT_TRUE(jpeg_file.is_open());
  jpeg_file.seekg(0, std::ios::end);
  jpeg_data.resize(jpeg_file.tellg());
  jpeg_file.seekg(0, std::ios::beg);
  jpeg_file.read(reinterpret_cast<char*>(jpeg_data.data()), jpeg_data.size());
  jpeg_file.close();

  ortc::Tensor<uint8_t> jpeg_tensor({static_cast<int64_t>(jpeg_data.size())},  jpeg_data.data());
  ortc::Tensor<uint8_t> out_tensor{&CppAllocator::Instance()};
  auto status = image_decoder.Compute(jpeg_tensor, out_tensor);
  ASSERT_TRUE(status.IsOk()) << status.ToString();

  ASSERT_EQ(out_tensor.Shape(), std::vector<int64_t>({876, 1300, 3}));
  auto out_range = out_tensor.Data() + 0;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({48, 14, 5, 48, 14, 5, 48, 14, 5, 48, 14, 5}));

#if OCOS_ENABLE_VENDOR_IMAGE_CODECS
  #if _WIN32
  out_range = out_tensor.Data() + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({228, 234, 222, 228, 235, 219, 219, 221, 200, 203, 201, 178}));

  out_range = out_tensor.Data() + 438 * 1300 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({84, 68, 53, 86, 70, 55, 92, 76, 60, 101, 86, 65}));

  out_range = out_tensor.Data() + 875 * 1300 * 3 + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({208, 210, 197, 204, 206, 193, 198, 200, 187, 194, 196, 183}));

  #elif __APPLE__
  out_range = out_tensor.Data() + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({225, 236, 222, 228, 235, 219, 218, 220, 199, 203, 201, 178}));

  out_range = out_tensor.Data() + 438 * 1300 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({84, 68, 53, 86, 70, 55, 92, 76, 59, 101, 86, 65}));

  out_range = out_tensor.Data() + 875 * 1300 * 3 + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({209, 211, 198, 204, 206, 193, 198, 200, 187, 194, 196, 183}));
  #else
  out_range = out_tensor.Data() + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({221, 237, 224, 225, 236, 219, 218, 222, 199, 203, 202, 174}));

  out_range = out_tensor.Data() + 438 * 1300 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({84, 68, 55, 86, 70, 55, 92, 77, 58, 101, 86, 65}));

  out_range = out_tensor.Data() + 875 * 1300 * 3 + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({208, 210, 197, 204, 206, 193, 198, 200, 187, 194, 196, 183}));
  #endif
#else
  out_range = out_tensor.Data() + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({221, 237, 224, 225, 236, 219, 218, 222, 199, 203, 202, 174}));

  out_range = out_tensor.Data() + 438 * 1300 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({84, 68, 55, 86, 70, 55, 92, 77, 58, 101, 86, 65}));

  out_range = out_tensor.Data() + 875 * 1300 * 3 + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({208, 210, 197, 204, 206, 193, 198, 200, 187, 194, 196, 183}));
#endif
}

#if OCOS_ENABLE_VENDOR_IMAGE_CODECS
#if defined(_WIN32) || defined(__APPLE__)
TEST(ImageDecoderTest, TestTiffDecoder) {
  ort_extensions::DecodeImage image_decoder;
  image_decoder.Init(std::unordered_map<std::string, std::variant<std::string>>());
  std::vector<uint8_t> tiff_data;
  std::filesystem::path tiff_path = "data/processor/canoe.tif";
  std::ifstream tiff_file(tiff_path, std::ios::binary);
  ASSERT_TRUE(tiff_file.is_open());
  tiff_file.seekg(0, std::ios::end);
  tiff_data.resize(tiff_file.tellg());
  tiff_file.seekg(0, std::ios::beg);
  tiff_file.read(reinterpret_cast<char*>(tiff_data.data()), tiff_data.size());
  tiff_file.close();

  ortc::Tensor<uint8_t> tiff_tensor({static_cast<int64_t>(tiff_data.size())},  tiff_data.data());
  ortc::Tensor<uint8_t> out_tensor{&CppAllocator::Instance()};
  auto status = image_decoder.Compute(tiff_tensor, out_tensor);
  ASSERT_TRUE(status.IsOk()) << status.ToString();

  ASSERT_EQ(out_tensor.Shape(), std::vector<int64_t>({207, 346, 3}));
  auto out_range = out_tensor.Data() + 0;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({66, 74, 57, 66, 74, 57, 66, 74, 57, 74, 66, 49}));

  out_range = out_tensor.Data() + 477 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({41, 41, 41, 33, 33, 33, 41, 41, 49, 33, 33, 33}));

  out_range = out_tensor.Data() + 103 * 346 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({24, 24, 24, 16, 16, 24, 16, 16, 24, 16, 16, 24}));

  out_range = out_tensor.Data() + 206 * 346 * 3 + 342 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({82, 66, 49, 74, 66, 57, 74, 66, 49, 82, 74, 57}));
}
#endif
#endif
