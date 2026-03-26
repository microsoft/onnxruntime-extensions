// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <tuple>
#include <fstream>
#include <filesystem>

#include "gtest/gtest.h"
#include "shared/api/c_api_utils.hpp"

#include "vision/decode_image.hpp"
#include "vision/encode_image.hpp"


using namespace ort_extensions;

TEST(ImgDecoderTest, TestPngEncoderDecoder) {
  ort_extensions::DecodeImage image_decoder;
  image_decoder.Init(std::unordered_map<std::string, std::variant<std::string>>());
  std::vector<uint8_t> png_data;
  std::filesystem::path png_path = "data/processor/exceltable.png";
  const size_t width = 487;
  const size_t height = 206;
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

  ASSERT_EQ(out_tensor.Shape(), std::vector<int64_t>({height, width, 3}));
  auto out_range = out_tensor.Data() + 0;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

  out_range = out_tensor.Data() + 477 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

  out_range = out_tensor.Data() + 243 * height * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217}));

  out_range = out_tensor.Data() + 485 * height * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

  ort_extensions::internal::EncodeImage image_encoder;
  image_encoder.OnInit();

  uint8_t* encodeOutputBuffer = nullptr;
  size_t encodeSize = 0;
  if (image_encoder.pngSupportsBgr()) {
    image_encoder.EncodePng(out_tensor.Data(), true, width, height, &encodeOutputBuffer, &encodeSize);
  } else {
    image_encoder.EncodePng(out_tensor.Data(), false, width, height, &encodeOutputBuffer, &encodeSize);
  }

  ASSERT_NE(encodeOutputBuffer, nullptr);
}

TEST(ImageDecoderTest, TestJpegEncoderDecoder) {
  ort_extensions::DecodeImage image_decoder;
  image_decoder.Init(std::unordered_map<std::string, std::variant<std::string>>());
  std::vector<uint8_t> jpeg_data;
  std::filesystem::path jpeg_path = "data/processor/australia.jpg";
  const size_t width = 1300;
  const size_t height = 876;
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

  ASSERT_EQ(out_tensor.Shape(), std::vector<int64_t>({height, width, 3}));
  auto out_range = out_tensor.Data() + 0;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({48, 14, 5, 48, 14, 5, 48, 14, 5, 48, 14, 5}));

#if OCOS_ENABLE_VENDOR_IMAGE_CODECS
  #if _WIN32
  out_range = out_tensor.Data() + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({228, 234, 222, 228, 235, 219, 219, 221, 200, 203, 201, 178}));

  out_range = out_tensor.Data() + 438 * width * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({84, 68, 53, 86, 70, 55, 92, 76, 60, 101, 86, 65}));

  out_range = out_tensor.Data() + 875 * width * 3 + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({208, 210, 197, 204, 206, 193, 198, 200, 187, 194, 196, 183}));

  #elif __APPLE__
  out_range = out_tensor.Data() + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({225, 236, 222, 225, 236, 222, 221, 219, 196, 203, 201, 178}));

  out_range = out_tensor.Data() + 438 * width * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({84, 68, 53, 86, 70, 55, 92, 77, 58, 101, 86, 67}));

  out_range = out_tensor.Data() + 875 * width * 3 + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({209, 211, 198, 204, 206, 193, 198, 200, 187, 194, 196, 183}));
  #else
  out_range = out_tensor.Data() + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({221, 237, 224, 225, 236, 219, 218, 222, 199, 203, 202, 174}));

  out_range = out_tensor.Data() + 438 * width * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({84, 68, 55, 86, 70, 55, 92, 77, 58, 101, 86, 65}));

  out_range = out_tensor.Data() + 875 * width * 3 + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({208, 210, 197, 204, 206, 193, 198, 200, 187, 194, 196, 183}));
  #endif
#else
  out_range = out_tensor.Data() + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({221, 237, 224, 225, 236, 219, 218, 222, 199, 203, 202, 174}));

  out_range = out_tensor.Data() + 438 * width * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({84, 68, 55, 86, 70, 55, 92, 77, 58, 101, 86, 65}));

  out_range = out_tensor.Data() + 875 * width * 3 + 1296 * 3;
  ASSERT_EQ(std::vector<uint8_t>(out_range, out_range + 12),
            std::vector<uint8_t>({208, 210, 197, 204, 206, 193, 198, 200, 187, 194, 196, 183}));
#endif

  ort_extensions::internal::EncodeImage image_encoder;
  image_encoder.OnInit();

  uint8_t* encodeOutputBuffer = nullptr;
  size_t encodeSize = 0;

  if (image_encoder.JpgSupportsBgr()) {
    image_encoder.EncodeJpg(out_tensor.Data(), true, width, height, &encodeOutputBuffer, &encodeSize);
  } else {
    image_encoder.EncodeJpg(out_tensor.Data(), false, width, height, &encodeOutputBuffer, &encodeSize);
  }

  ASSERT_NE(encodeOutputBuffer, nullptr);
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
// Security: verify that oversized PNG images are rejected (decompression bomb mitigation).
// Crafts a minimal valid PNG with an IHDR claiming 20000x20000 dimensions.
TEST(ImageDecoderTest, TestPngOversizeDimensionsRejected) {
  ort_extensions::DecodeImage image_decoder;
  image_decoder.Init(std::unordered_map<std::string, std::variant<std::string>>());

  // Minimal PNG: signature + IHDR (20000x20000) + IDAT (minimal zlib) + IEND.
  // png_read_info reads all chunks up to the first IDAT, so we need IDAT present
  // for libpng to successfully parse the header and reach our dimension check.
  std::vector<uint8_t> png_oversize = {
    // PNG signature
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,

    // IHDR chunk: length=13
    0x00, 0x00, 0x00, 0x0D,
    // "IHDR"
    0x49, 0x48, 0x44, 0x52,
    // Width = 20000 (0x00004E20)
    0x00, 0x00, 0x4E, 0x20,
    // Height = 20000 (0x00004E20)
    0x00, 0x00, 0x4E, 0x20,
    // Bit depth = 8, Color type = 2 (RGB), Compression = 0, Filter = 0, Interlace = 0
    0x08, 0x02, 0x00, 0x00, 0x00,
    // CRC32 of IHDR
    0x6C, 0x12, 0xD1, 0x6E,

    // IDAT chunk: length=11 (minimal valid zlib stream)
    0x00, 0x00, 0x00, 0x0B,
    // "IDAT"
    0x49, 0x44, 0x41, 0x54,
    // Minimal zlib: header(78 01) + stored block BFINAL=1 LEN=0 NLEN=FFFF + Adler32(00000001)
    0x78, 0x01, 0x01, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01,
    // CRC32 of IDAT type + data
    0x89, 0xD6, 0xAE, 0x5F,

    // IEND chunk: length=0
    0x00, 0x00, 0x00, 0x00,
    // "IEND"
    0x49, 0x45, 0x4E, 0x44,
    // CRC32 of IEND
    0xAE, 0x42, 0x60, 0x82
  };

  ortc::Tensor<uint8_t> png_tensor({static_cast<int64_t>(png_oversize.size())}, png_oversize.data());
  ortc::Tensor<uint8_t> out_tensor{&CppAllocator::Instance()};
  auto status = image_decoder.Compute(png_tensor, out_tensor);

  // Must be rejected by the dimension check, not just any decode failure.
  std::cout << "[Expected rejection] PNG 20000x20000: " << status.ToString() << std::endl;
  ASSERT_FALSE(status.IsOk()) << "Oversized PNG (20000x20000) should have been rejected but was accepted.";
  ASSERT_NE(status.ToString().find("dimensions exceed"), std::string::npos)
      << "Expected dimension-limit error, got: " << status.ToString();
}

// Security: verify that oversized JPEG images are rejected (decompression bomb mitigation).
// Crafts a minimal JPEG with SOF0 claiming 17000x17000 dimensions.
TEST(ImageDecoderTest, TestJpegOversizeDimensionsRejected) {
  ort_extensions::DecodeImage image_decoder;
  image_decoder.Init(std::unordered_map<std::string, std::variant<std::string>>());

  // Minimal JPEG: SOI + SOF0 (with oversized dimensions) + EOI
  // This should be enough for jpeg_read_header + jpeg_start_decompress to parse dimensions.
  // Craft a minimal but structurally valid JPEG so libjpeg can parse the header
  // and reach jpeg_start_decompress where our dimension check fires.
  // Structure: SOI + DQT + SOF0 (oversized dims) + SOS + EOI
  std::vector<uint8_t> jpeg_oversize = {
    // SOI
    0xFF, 0xD8,

    // DQT marker (required for jpeg_start_decompress to succeed)
    0xFF, 0xDB,
    // Length = 67 (2 + 1 + 64): precision/table byte + 64 quantization values
    0x00, 0x43,
    // Table 0, 8-bit precision
    0x00,
    // 64 quantization values (all 1s — minimal valid table)
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,

    // SOF0 marker
    0xFF, 0xC0,
    // Length = 11
    0x00, 0x0B,
    // Precision = 8
    0x08,
    // Height = 17000 (0x4268)
    0x42, 0x68,
    // Width = 17000 (0x4268)
    0x42, 0x68,
    // Number of components = 1 (grayscale)
    0x01,
    // Component 1: id=1, sampling=0x11, quant_table=0
    0x01, 0x11, 0x00,

    // DHT marker (minimal Huffman table for DC, required by libjpeg)
    0xFF, 0xC4,
    // Length = 31 (2 + 1 + 16 + 12 symbols)
    0x00, 0x1F,
    // DC table 0
    0x00,
    // Number of codes of each length 1-16 (12 codes total)
    0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // Symbol values
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B,

    // SOS marker
    0xFF, 0xDA,
    // Length = 8
    0x00, 0x08,
    // Number of components = 1
    0x01,
    // Component 1: DC table 0, AC table 0
    0x01, 0x00,
    // Spectral selection start, end, approximation
    0x00, 0x3F, 0x00,

    // Minimal scan data (a single zero byte) + EOI
    0x00,
    0xFF, 0xD9
  };

  ortc::Tensor<uint8_t> jpeg_tensor({static_cast<int64_t>(jpeg_oversize.size())}, jpeg_oversize.data());
  ortc::Tensor<uint8_t> out_tensor{&CppAllocator::Instance()};
  auto status = image_decoder.Compute(jpeg_tensor, out_tensor);

  // Must be rejected by the dimension check, not just any decode failure.
  std::cout << "[Expected rejection] JPEG 17000x17000: " << status.ToString() << std::endl;
  ASSERT_FALSE(status.IsOk()) << "Oversized JPEG (17000x17000) should have been rejected but was accepted.";
  ASSERT_NE(status.ToString().find("dimensions exceed"), std::string::npos)
      << "Expected dimension-limit error, got: " << status.ToString();
}