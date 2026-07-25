// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <csetjmp>
#include <cstdint>
#include <string>

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

  struct JpegErrorManager {
    jpeg_error_mgr base;
    jmp_buf jump_buffer;
    char message[JMSG_LENGTH_MAX]{};

    static void ErrorExit(j_common_ptr cinfo) {
      auto* error = reinterpret_cast<JpegErrorManager*>(cinfo->err);
      (*cinfo->err->format_message)(cinfo, error->message);
      longjmp(error->jump_buffer, 1);
    }
  };

  class JMemorySourceManager : public jpeg_source_mgr {
   public:
    JMemorySourceManager(const uint8_t* encoded_image_data, const int64_t encoded_image_data_len) {
      next_input_byte = reinterpret_cast<const JOCTET*>(encoded_image_data);
      bytes_in_buffer = static_cast<size_t>(encoded_image_data_len);
      init_source = &JMemorySourceManager::initSource;
      fill_input_buffer = &JMemorySourceManager::fillInputBuffer;
      skip_input_data = &JMemorySourceManager::skipInputData;
      resync_to_restart = jpeg_resync_to_restart;
      term_source = &JMemorySourceManager::termSource;
    }

    static void initSource(j_decompress_ptr cinfo) {
      // No initialization needed
    }

    // This is an in-memory, non-suspending source. Asking for more bytes means
    // the JPEG is truncated, so report a fatal libjpeg error immediately.
    static boolean fillInputBuffer(j_decompress_ptr cinfo) {
      auto* srcMgr = reinterpret_cast<JMemorySourceManager*>(cinfo->src);
      srcMgr->extError = kOrtxErrorCorruptData;
      ERREXIT(cinfo, JERR_INPUT_EOF);
      return FALSE;
    }

    static void skipInputData(j_decompress_ptr cinfo, long num_bytes) {
      auto* srcMgr = reinterpret_cast<JMemorySourceManager*>(cinfo->src);
      if (num_bytes > 0) {
        size_t bytes_to_skip = static_cast<size_t>(num_bytes);
        if (bytes_to_skip > srcMgr->bytes_in_buffer) {
          srcMgr->next_input_byte += srcMgr->bytes_in_buffer;
          srcMgr->bytes_in_buffer = 0;
          srcMgr->extError = kOrtxErrorCorruptData;
          ERREXIT(cinfo, JERR_INPUT_EOF);
          return;
        }
        srcMgr->next_input_byte += bytes_to_skip;
        srcMgr->bytes_in_buffer -= bytes_to_skip;
      }
    }

    static void termSource(j_decompress_ptr cinfo) {
      // No cleanup needed
    }

    extError_t extError{kOrtxOK};
  };

  // libjpeg mutates its state after setjmp. Keep that mutable state on the
  // heap so automatic variables do not become indeterminate after longjmp.
  struct JpegDecodeState {
    JpegDecodeState(const uint8_t* data, int64_t size) : source(data, size) {}

    jpeg_decompress_struct cinfo{};
    JpegErrorManager error{};
    JMemorySourceManager source;
    std::vector<int64_t> output_dimensions;
  };

  static void DestroyJpegState(JpegDecodeState* state) {
    // jpeg_create_decompress initializes mem to null before doing work, so this
    // also cleans up a partially created decompressor.
    if (state->cinfo.mem != nullptr) {
      jpeg_destroy_decompress(&state->cinfo);
    }
    delete state;
  }

  static OrtxStatus JpegFailure(JpegDecodeState* state, const std::string& message) {
    DestroyJpegState(state);
    return {kOrtxErrorCorruptData, message};
  }

  OrtxStatus DecodePNG(const uint8_t* encoded_image_data, const int64_t encoded_image_data_len,
                       ortc::Tensor<uint8_t>& output) const {
    // Decode the PNG image
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
      return {kOrtxErrorCorruptData, "[ImageDecoder]: Failed to create png read struct."};
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
      png_destroy_read_struct(&png, nullptr, nullptr);
      return {kOrtxErrorCorruptData, "[ImageDecoder]: Failed to create png info struct."};
    }

    if (setjmp(png_jmpbuf(png))) {
      png_destroy_read_struct(&png, &info, nullptr);
      return {kOrtxErrorCorruptData, "[ImageDecoder]: Error during png creation."};
    }

    struct BufferState {
      const uint8_t* ptr;
      png_size_t size;
    } bufferState = {encoded_image_data, static_cast<png_size_t>(encoded_image_data_len)};

    png_set_read_fn(png, &bufferState, [](png_structp pngPtr, png_bytep data, png_size_t length) {
      BufferState* state = static_cast<BufferState*>(png_get_io_ptr(pngPtr));
      if (length > state->size) png_error(pngPtr, "Read Error: Exceeded buffer size");
      memcpy(data, state->ptr, length);
      state->ptr += length;
      state->size -= length;
    });

    png_read_info(png, info);

    auto width = png_get_image_width(png, info);
    auto height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

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

    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE) {
      png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
      png_set_gray_to_rgb(png);
    }

    png_read_update_info(png, info);

    // Dimension limit to prevent decompression bombs
    if (width > kMaxImageDimension || height > kMaxImageDimension ||
        static_cast<uint64_t>(width) * height > kMaxPixelCount) {
      png_destroy_read_struct(&png, &info, nullptr);
      return {kOrtxErrorInvalidArgument,
              "[ImageDecoder]: PNG dimensions exceed maximum allowed size."};
    }

    std::vector<int64_t> output_dimensions{height, width, 3};
    uint8_t* output_data = output.Allocate(output_dimensions);
    // Read the image row by row
    std::vector<uint8_t> row(width * 4);
    for (uint32_t i = 0; i < height; ++i) {
      png_read_row(png, row.data(), nullptr);
      for (uint32_t j = 0; j < width; ++j) {
        for (uint32_t k = 0; k < 3; ++k) {
          output_data[i * width * 3 + j * 3 + k] = row[j * 4 + k];
        }
      }
    }

    png_destroy_read_struct(&png, &info, nullptr);
    return {};
  }

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) const {
    const auto& dimensions = input.Shape();
    if (dimensions.size() != 1ULL) {
      return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Only raw image formats are supported."};
    }

    // Get data & the length
    const uint8_t* encoded_image_data = input.Data();
    const int64_t encoded_image_data_len = input.NumberOfElement();

    // check it's a PNG image or JPEG image
    if (encoded_image_data_len < 8) {
      return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Invalid image data."};
    }

    if (png_sig_cmp(encoded_image_data, 0, 8) == 0) {
      return DecodePNG(encoded_image_data, encoded_image_data_len, output);
    } else {
      auto* const state =
          new JpegDecodeState(encoded_image_data, encoded_image_data_len);
      state->cinfo.err = jpeg_std_error(&state->error.base);
      state->error.base.error_exit = &JpegErrorManager::ErrorExit;

      if (setjmp(state->error.jump_buffer)) {
        const char* diagnostic =
            state->error.message[0] == '\0'
                ? "unknown libjpeg error"
                : state->error.message;
        return JpegFailure(
            state,
            std::string("[ImageDecoder]: Failed to decode JPEG image: ") + diagnostic);
      }

      jpeg_create_decompress(&state->cinfo);
      state->cinfo.src = &state->source;

      // Read the JPEG header to get image info
      if (jpeg_read_header(&state->cinfo, TRUE) != JPEG_HEADER_OK) {
        return JpegFailure(
            state, "[ImageDecoder]: Failed to decode JPEG image header.");
      }

      // Security: explicitly reject CMYK/YCCK color spaces before decompression.
      // These have 4 channels and downstream code assumes 3 channels (CVE-class: CWE-122).
      if (state->cinfo.jpeg_color_space == JCS_CMYK ||
          state->cinfo.jpeg_color_space == JCS_YCCK) {
        DestroyJpegState(state);
        return {kOrtxErrorInvalidArgument,
                "[ImageDecoder]: Unsupported JPEG color space (CMYK/YCCK). Only RGB and grayscale are supported."};
      }

      // Force RGB output to ensure consistent 3-channel output regardless of input
      // (e.g., grayscale JPEGs are expanded to RGB).
      state->cinfo.out_color_space = JCS_RGB;

      // Start decompression
      if (!jpeg_start_decompress(&state->cinfo)) {
        return JpegFailure(
            state, "[ImageDecoder]: Failed to start JPEG decompression.");
      }

      // Dimension limit to prevent decompression bombs
      if (state->cinfo.output_width > kMaxImageDimension ||
          state->cinfo.output_height > kMaxImageDimension ||
          static_cast<uint64_t>(state->cinfo.output_width) *
                  state->cinfo.output_height >
              kMaxPixelCount) {
        DestroyJpegState(state);
        return {kOrtxErrorInvalidArgument,
                "[ImageDecoder]: JPEG dimensions exceed maximum allowed size."};
      }

      // Safety net: verify 3-channel output after decompression.
      if (state->cinfo.output_components != 3) {
        const int output_components = state->cinfo.output_components;
        DestroyJpegState(state);
        return {kOrtxErrorInvalidArgument,
                "[ImageDecoder]: Unexpected JPEG output channels. Expected 3 (RGB), got " +
                std::to_string(output_components) + "."};
      }

      // Allocate memory for the image
      state->output_dimensions = {
          state->cinfo.output_height,
          state->cinfo.output_width,
          state->cinfo.output_components};
      uint8_t* imageBuffer = output.Allocate(state->output_dimensions);

      // Read the image data
      int row_stride =
          state->cinfo.output_width * state->cinfo.output_components;
      while (state->cinfo.output_scanline < state->cinfo.output_height) {
        uint8_t* row_ptr =
            imageBuffer + (state->cinfo.output_scanline * row_stride);
        if (jpeg_read_scanlines(&state->cinfo, &row_ptr, 1) != 1) {
          state->source.extError = kOrtxErrorCorruptData;
          break;
        }
      }

      if (state->source.extError != kOrtxOK) {
        return JpegFailure(
            state, "[ImageDecoder]: Failed to decode JPEG image.");
      }

      // Finish decompression
      if (!jpeg_finish_decompress(&state->cinfo)) {
        return JpegFailure(
            state, "[ImageDecoder]: Failed to finish JPEG decompression.");
      }
      DestroyJpegState(state);
    }
    return {};
  }
};

}  // namespace ort_extensions::internal
