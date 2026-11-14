// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "png.h"
#if _WIN32
// Fix redefinition in jmorecfg.h
#include <basetsd.h>
#endif
#include "jpeglib.h"
#include "jerror.h"
#include "op_def_struct.h"
#include "ext_status.h"

namespace ort_extensions::internal {

// Maximum image dimension (width or height) and total pixel count to prevent decompression bombs.
static constexpr uint64_t kMaxImageDimension = 16384;
static constexpr uint64_t kMaxPixelCount = 100'000'000;  // 100 megapixels

struct DecodeImage {
  OrtxStatus OnInit() { return {}; }

  OrtxStatus DecodePNG(const uint8_t* data, int64_t size, ortc::Tensor<uint8_t>& output) const {
    if (data == nullptr || size <= 0) {
      return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Invalid image data."};
    }

    PngDecoder decoder(data, static_cast<size_t>(size));
    return decoder.Read(output);
  }

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) const {
    if (input.Shape().size() != 1ULL) {
      return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Only raw image formats are supported."};
    }

    const auto* data = input.Data();
    const auto size = input.NumberOfElement();
    if (data == nullptr || size < 8) {
      return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Invalid image data."};
    }

    if (png_sig_cmp(data, 0, 8) == 0) {
      return DecodePNG(data, size, output);
    }

    JpegDecoder decoder(data, static_cast<size_t>(size));
    return decoder.Read(output);
  }

 private:
  static bool ValidDimensions(uint64_t width, uint64_t height) {
    return width != 0 && height != 0 && width <= kMaxImageDimension && height <= kMaxImageDimension &&
           width * height <= kMaxPixelCount;
  }

  // Keep mutable codec state and C++ owners in the caller of Read: longjmp must not
  // bypass destructors or leave modified automatic objects in the setjmp frame.
  struct PngDecoder {
    const uint8_t* data;
    size_t remaining;
    png_structp png{nullptr};
    png_infop info{nullptr};
    std::vector<png_bytep> rows;

    PngDecoder(const uint8_t* input, size_t size) : data(input), remaining(size) {}
    PngDecoder(const PngDecoder&) = delete;
    PngDecoder& operator=(const PngDecoder&) = delete;

    ~PngDecoder() {
      if (png != nullptr) {
        png_destroy_read_struct(&png, &info, nullptr);
      }
    }

    void AllocateRows(ortc::Tensor<uint8_t>& output, png_uint_32 width, png_uint_32 height) {
      auto* pixels = output.Allocate({height, width, 3});
      rows.resize(height);
      for (size_t y = 0; y < height; ++y) {
        rows[y] = pixels + y * width * 3;
      }
    }

    OrtxStatus Read(ortc::Tensor<uint8_t>& output) {
      png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
      if (png == nullptr) {
        return {kOrtxErrorCorruptData, "[ImageDecoder]: Failed to create png read struct."};
      }

      info = png_create_info_struct(png);
      if (info == nullptr) {
        return {kOrtxErrorCorruptData, "[ImageDecoder]: Failed to create png info struct."};
      }

      if (setjmp(png_jmpbuf(png))) {
        return {kOrtxErrorCorruptData, "[ImageDecoder]: Failed to decode PNG image."};
      }

      png_set_read_fn(png, this, [](png_structp reader, png_bytep destination, png_size_t length) {
        auto& state = *static_cast<PngDecoder*>(png_get_io_ptr(reader));
        if (length > state.remaining) {
          png_error(reader, "Read Error: Exceeded buffer size");
        }

        std::memcpy(destination, state.data, length);
        state.data += length;
        state.remaining -= length;
      });

      png_read_info(png, info);
      const auto width = png_get_image_width(png, info);
      const auto height = png_get_image_height(png, info);
      if (!ValidDimensions(width, height)) {
        return {kOrtxErrorInvalidArgument, "[ImageDecoder]: PNG dimensions exceed maximum allowed size."};
      }

      const auto color_type = png_get_color_type(png, info);
      const auto bit_depth = png_get_bit_depth(png, info);
      if (bit_depth == 16) {
        png_set_strip_16(png);
      }

      if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
      }

      if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
      }

      if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
      }

      png_set_strip_alpha(png);
      if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png);
      }

      png_set_interlace_handling(png);
      png_read_update_info(png, info);
      if (png_get_bit_depth(png, info) != 8 || png_get_channels(png, info) != 3 ||
          png_get_rowbytes(png, info) != static_cast<size_t>(width) * 3) {
        return {kOrtxErrorCorruptData, "[ImageDecoder]: Unexpected PNG output layout."};
      }

      AllocateRows(output, width, height);
      png_read_image(png, rows.data());
      png_read_end(png, info);
      return {};
    }
  };

  struct JpegErrorManager : jpeg_error_mgr {
    std::jmp_buf jump;
    char message[JMSG_LENGTH_MAX]{};

    static void ErrorExit(j_common_ptr cinfo) {
      auto& error = *static_cast<JpegErrorManager*>(cinfo->err);
      error.format_message(cinfo, error.message);
      std::longjmp(error.jump, 1);
    }

    static void EmitMessage(j_common_ptr cinfo, int level) {
      // libjpeg otherwise recovers from premature scan ends with fabricated pixels.
      if (level < 0) {
        ErrorExit(cinfo);
      }
    }
  };

  struct JMemorySourceManager : jpeg_source_mgr {
    JMemorySourceManager(const uint8_t* data, size_t size) {
      next_input_byte = reinterpret_cast<const JOCTET*>(data);
      bytes_in_buffer = size;
      init_source = [](j_decompress_ptr) {};
      fill_input_buffer = [](j_decompress_ptr cinfo) -> boolean {
        // This is a complete in-memory image, not a suspendable stream.
        ERREXIT(cinfo, JERR_INPUT_EOF);
        return FALSE;
      };
      skip_input_data = [](j_decompress_ptr cinfo, long count) {
        if (count <= 0) {
          return;
        }

        auto& source = *cinfo->src;
        const auto size_to_skip = static_cast<size_t>(count);
        if (size_to_skip > source.bytes_in_buffer) {
          ERREXIT(cinfo, JERR_INPUT_EOF);
        }

        source.next_input_byte += size_to_skip;
        source.bytes_in_buffer -= size_to_skip;
      };
      resync_to_restart = jpeg_resync_to_restart;
      term_source = [](j_decompress_ptr) {};
    }
  };

  struct JpegDecoder {
    jpeg_decompress_struct cinfo{};
    JpegErrorManager error{};
    JMemorySourceManager source;

    JpegDecoder(const uint8_t* data, size_t size) : source(data, size) {
      cinfo.err = jpeg_std_error(&error);
      error.error_exit = JpegErrorManager::ErrorExit;
      error.emit_message = JpegErrorManager::EmitMessage;
    }

    JpegDecoder(const JpegDecoder&) = delete;
    JpegDecoder& operator=(const JpegDecoder&) = delete;

    ~JpegDecoder() { jpeg_destroy_decompress(&cinfo); }

    static uint8_t* AllocateOutput(ortc::Tensor<uint8_t>& output, const jpeg_decompress_struct& info) {
      return output.Allocate({info.output_height, info.output_width, 3});
    }

    OrtxStatus Read(ortc::Tensor<uint8_t>& output) {
      if (setjmp(error.jump)) {
        return {kOrtxErrorCorruptData, std::string("[ImageDecoder]: Failed to decode JPEG image: ") + error.message};
      }

      jpeg_create_decompress(&cinfo);
      cinfo.src = &source;
      if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        return {kOrtxErrorCorruptData, "[ImageDecoder]: Incomplete JPEG header."};
      }

      if (cinfo.jpeg_color_space == JCS_CMYK || cinfo.jpeg_color_space == JCS_YCCK) {
        return {kOrtxErrorInvalidArgument,
                "[ImageDecoder]: Unsupported JPEG color space (CMYK/YCCK). Only RGB and grayscale are supported."};
      }

      // Check the header before starting decompression, which can allocate image-sized buffers.
      if (!ValidDimensions(cinfo.image_width, cinfo.image_height)) {
        return {kOrtxErrorInvalidArgument, "[ImageDecoder]: JPEG dimensions exceed maximum allowed size."};
      }

      cinfo.out_color_space = JCS_RGB;
      if (!jpeg_start_decompress(&cinfo)) {
        return {kOrtxErrorCorruptData, "[ImageDecoder]: Incomplete JPEG scan."};
      }

      if (!ValidDimensions(cinfo.output_width, cinfo.output_height) || cinfo.output_components != 3) {
        return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Unexpected JPEG output dimensions or channels."};
      }

      auto* pixels = AllocateOutput(output, cinfo);
      const auto stride = static_cast<size_t>(cinfo.output_width) * 3;
      while (cinfo.output_scanline < cinfo.output_height) {
        auto* row = pixels + static_cast<size_t>(cinfo.output_scanline) * stride;
        if (jpeg_read_scanlines(&cinfo, &row, 1) != 1) {
          return {kOrtxErrorCorruptData, "[ImageDecoder]: Incomplete JPEG scanline."};
        }
      }

      if (!jpeg_finish_decompress(&cinfo)) {
        return {kOrtxErrorCorruptData, "[ImageDecoder]: Incomplete JPEG image."};
      }

      return {};
    }
  };
};

}  // namespace ort_extensions::internal
