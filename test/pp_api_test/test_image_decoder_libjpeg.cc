// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Exercise the portable decoder even on platforms whose default is a vendor codec.
// It is only linked when the portable codecs are enabled.
#if !OCOS_ENABLE_VENDOR_IMAGE_CODECS || (!defined(_WIN32) && !defined(__APPLE__))

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "shared/api/c_api_utils.hpp"
#include "operators/vision/image_decoder.hpp"

namespace {

using ort_extensions::CppAllocator;
using ort_extensions::internal::DecodeImage;

std::vector<uint8_t> MakeJpeg(J_COLOR_SPACE color = JCS_RGB, bool progressive = false) {
  jpeg_compress_struct info{};
  jpeg_error_mgr error{};
  info.err = jpeg_std_error(&error);
  jpeg_create_compress(&info);
  unsigned char* buffer = nullptr;
  size_t size = 0;
  jpeg_mem_dest(&info, &buffer, &size);
  info.image_width = 16;
  info.image_height = 16;
  info.input_components = color == JCS_GRAYSCALE ? 1 : (color == JCS_CMYK ? 4 : 3);
  info.in_color_space = color;
  jpeg_set_defaults(&info);
  if (progressive) {
    jpeg_simple_progression(&info);
  }

  jpeg_start_compress(&info, TRUE);
  std::vector<uint8_t> row(16 * info.input_components, 42);
  while (info.next_scanline < info.image_height) {
    auto* pixels = row.data();
    jpeg_write_scanlines(&info, &pixels, 1);
  }

  jpeg_finish_compress(&info);
  std::unique_ptr<unsigned char, decltype(&std::free)> owner(buffer, &std::free);
  std::vector<uint8_t> result(buffer, buffer + size);
  jpeg_destroy_compress(&info);
  return result;
}

std::vector<uint8_t> MakePng(int color = PNG_COLOR_TYPE_RGB, bool interlaced = false, int depth = 8) {
  std::vector<uint8_t> result;
  auto* png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  auto* info = png_create_info_struct(png);
  png_set_write_fn(
      png, &result,
      [](png_structp writer, png_bytep data, png_size_t size) {
        auto& bytes = *static_cast<std::vector<uint8_t>*>(png_get_io_ptr(writer));
        bytes.insert(bytes.end(), data, data + size);
      },
      [](png_structp) {});
  png_set_IHDR(png, info, 16, 16, depth, color, interlaced ? PNG_INTERLACE_ADAM7 : PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_color palette[] = {{42, 42, 42}};
  png_byte alpha[] = {128};
  if (color == PNG_COLOR_TYPE_PALETTE) {
    png_set_PLTE(png, info, palette, 1);
    png_set_tRNS(png, info, alpha, 1, nullptr);
  }

  png_write_info(png, info);
  const auto channels = png_get_channels(png, info);
  std::vector<uint8_t> row(16 * channels * (depth / 8), color == PNG_COLOR_TYPE_PALETTE ? 0 : 42);
  std::vector<png_bytep> rows(16, row.data());
  png_write_image(png, rows.data());
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  return result;
}

OrtxStatus Decode(const DecodeImage& decoder, const std::vector<uint8_t>& bytes, ortc::Tensor<uint8_t>& output) {
  ortc::Tensor<uint8_t> input({static_cast<int64_t>(bytes.size())}, const_cast<uint8_t*>(bytes.data()));
  return decoder.Compute(input, output);
}

void ExpectSolidImage(const DecodeImage& decoder, const std::vector<uint8_t>& bytes) {
  ortc::Tensor<uint8_t> output{&CppAllocator::Instance()};
  const auto status = Decode(decoder, bytes, output);
  ASSERT_TRUE(status.IsOk()) << status.ToString();
  ASSERT_EQ(output.Shape(), std::vector<int64_t>({16, 16, 3}));
  EXPECT_EQ(std::vector<uint8_t>(output.Data(), output.Data() + 16 * 16 * 3), std::vector<uint8_t>(16 * 16 * 3, 42));
}

TEST(PortableImageDecoderTest, ValidRgbGrayscaleAndProgressiveJpegHaveThreeChannels) {
  const DecodeImage decoder;
  ExpectSolidImage(decoder, MakeJpeg());
  ExpectSolidImage(decoder, MakeJpeg(JCS_GRAYSCALE));
  ExpectSolidImage(decoder, MakeJpeg(JCS_RGB, true));
}

TEST(PortableImageDecoderTest, PngColorTransformsAndInterlacingPreserveRgbPixels) {
  const DecodeImage decoder;
  for (const auto color : {PNG_COLOR_TYPE_RGB, PNG_COLOR_TYPE_RGB_ALPHA, PNG_COLOR_TYPE_GRAY, PNG_COLOR_TYPE_GRAY_ALPHA,
                           PNG_COLOR_TYPE_PALETTE}) {
    for (const auto interlaced : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "color=" << color << " interlaced=" << interlaced);
      ExpectSolidImage(decoder, MakePng(color, interlaced));
    }
  }

  ExpectSolidImage(decoder, MakePng(PNG_COLOR_TYPE_RGB, false, 16));
}

TEST(PortableImageDecoderTest, MalformedJpegReturnsCorruptDataAndDecoderRecovers) {
  const DecodeImage decoder;
  const auto valid = MakeJpeg();
  for (const auto& malformed : std::vector<std::vector<uint8_t>>{std::vector<uint8_t>(8, 0),
                                                                 {0xff, 0xd8, 0xff, 0xe1, 0xff, 0xff, 0, 0},
                                                                 {0xff, 0xd8, 0xff, 0xc0, 0, 2, 0, 0}}) {
    ortc::Tensor<uint8_t> output{&CppAllocator::Instance()};
    const auto status = Decode(decoder, malformed, output);
    EXPECT_EQ(status.Code(), kOrtxErrorCorruptData) << status.ToString();
    EXPECT_NE(std::string(status.Message()).find("JPEG"), std::string::npos);
    ExpectSolidImage(decoder, valid);
  }
}

TEST(PortableImageDecoderTest, EveryTruncatedJpegPrefixIsRejectedAndDecoderRecovers) {
  const DecodeImage decoder;
  for (const auto progressive : {false, true}) {
    const auto valid = MakeJpeg(JCS_RGB, progressive);
    for (size_t size = 0; size < valid.size(); ++size) {
      SCOPED_TRACE(::testing::Message() << "prefix=" << size << " progressive=" << progressive);
      ortc::Tensor<uint8_t> output{&CppAllocator::Instance()};
      const auto status = Decode(decoder, {valid.begin(), valid.begin() + size}, output);
      EXPECT_EQ(status.Code(), size < 8 ? kOrtxErrorInvalidArgument : kOrtxErrorCorruptData);
    }

    ExpectSolidImage(decoder, valid);
  }
}

TEST(PortableImageDecoderTest, PrematureJpegEndMarkerCannotHideMissingScanPayload) {
  const DecodeImage decoder;
  const auto valid = MakeJpeg();
  const uint8_t scan_marker[] = {0xff, 0xda};
  const auto scan = std::search(valid.begin(), valid.end(), std::begin(scan_marker), std::end(scan_marker));
  ASSERT_NE(scan, valid.end());
  const auto scan_offset = static_cast<size_t>(scan - valid.begin());
  const auto payload = scan_offset + 2 + (valid[scan_offset + 2] << 8) + valid[scan_offset + 3];
  ASSERT_LT(payload, valid.size() - 2);
  for (size_t size = payload; size < valid.size() - 2; ++size) {
    SCOPED_TRACE(size);
    std::vector<uint8_t> truncated(valid.begin(), valid.begin() + size);
    truncated.insert(truncated.end(), {0xff, 0xd9});
    ortc::Tensor<uint8_t> output{&CppAllocator::Instance()};
    EXPECT_EQ(Decode(decoder, truncated, output).Code(), kOrtxErrorCorruptData);
  }

  ExpectSolidImage(decoder, valid);
}

TEST(PortableImageDecoderTest, EveryTruncatedPngPrefixIsRejectedAndDecoderRecovers) {
  const DecodeImage decoder;
  const auto valid = MakePng();
  for (size_t size = 0; size < valid.size(); ++size) {
    SCOPED_TRACE(size);
    ortc::Tensor<uint8_t> output{&CppAllocator::Instance()};
    const auto status = Decode(decoder, {valid.begin(), valid.begin() + size}, output);
    EXPECT_EQ(status.Code(), size < 8 ? kOrtxErrorInvalidArgument : kOrtxErrorCorruptData);
  }

  ExpectSolidImage(decoder, valid);
}

TEST(PortableImageDecoderTest, CmykJpegIsRejectedBeforeAllocatingOutput) {
  const DecodeImage decoder;
  ortc::Tensor<uint8_t> output{&CppAllocator::Instance()};
  EXPECT_EQ(Decode(decoder, MakeJpeg(JCS_CMYK), output).Code(), kOrtxErrorInvalidArgument);
  EXPECT_EQ(output.Data(), nullptr);
}

TEST(PortableImageDecoderTest, OversizedProgressiveJpegIsRejectedBeforeAllocatingOutput) {
  const DecodeImage decoder;
  auto bytes = MakeJpeg(JCS_RGB, true);
  const uint8_t frame_marker[] = {0xff, 0xc2};
  const auto frame = std::search(bytes.begin(), bytes.end(), std::begin(frame_marker), std::end(frame_marker));
  ASSERT_NE(frame, bytes.end());
  // SOF height and width follow the marker, length, and precision.
  std::fill(frame + 5, frame + 9, 0x7f);
  ortc::Tensor<uint8_t> output{&CppAllocator::Instance()};
  const auto status = Decode(decoder, bytes, output);
  EXPECT_EQ(status.Code(), kOrtxErrorInvalidArgument);
  EXPECT_NE(std::string(status.Message()).find("dimensions"), std::string::npos);
  EXPECT_EQ(output.Data(), nullptr);
}

TEST(PortableImageDecoderTest, InvalidRankAndNullPngInputReturnInvalidArgument) {
  const DecodeImage decoder;
  ortc::Tensor<uint8_t> output{&CppAllocator::Instance()};
  std::vector<uint8_t> bytes(8);
  ortc::Tensor<uint8_t> input({2, 4}, bytes.data());
  EXPECT_EQ(decoder.Compute(input, output).Code(), kOrtxErrorInvalidArgument);
  EXPECT_EQ(decoder.DecodePNG(nullptr, 8, output).Code(), kOrtxErrorInvalidArgument);
  EXPECT_EQ(decoder.DecodePNG(bytes.data(), -1, output).Code(), kOrtxErrorInvalidArgument);
}

}  // namespace

#endif
