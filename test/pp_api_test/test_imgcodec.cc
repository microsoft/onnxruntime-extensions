// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <tuple>
#include <fstream>
#include <filesystem>

#include "gtest/gtest.h"
#include "shared/api/c_api_utils.hpp"
#include "shared/api/image_decoder.hpp"

using namespace ort_extensions;

TEST(ImgDecoderTest, TestPngDecoder) {
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
  auto status = image_decoder(png_tensor, out_tensor);
  ASSERT_TRUE(status.IsOk());

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
  auto status = image_decoder(jpeg_tensor, out_tensor);
  ASSERT_TRUE(status.IsOk());

  ASSERT_EQ(out_tensor.Shape(), std::vector<int64_t>({876, 1300, 3}));
  auto out_range = out_tensor.Data() + 0;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({48, 14, 5, 48, 14, 5, 48, 14, 5, 48, 14, 5}));

  out_range = out_tensor.Data() + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({221, 237, 224, 225, 236, 219, 218, 222, 199, 203, 202, 174}));

  out_range = out_tensor.Data() + 438 * 1300 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({84, 68, 55, 86, 70, 55, 92, 77, 58, 101, 86, 65}));

  out_range = out_tensor.Data() + 875 * 1300 * 3 + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({208, 210, 197, 204, 206, 193, 198, 200, 187, 194, 196, 183}));
}
