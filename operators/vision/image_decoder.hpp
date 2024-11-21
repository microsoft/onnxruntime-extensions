// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "png.h"
#if _WIN32
// Fix redefinition in jmorecfg.h
#include <basetsd.h>
#endif
#include "jpeglib.h"
#include "op_def_struct.h"
#include "ext_status.h"

namespace ort_extensions::internal {
struct DecodeImage {
  OrtxStatus OnInit() { return {}; }

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
      // Initialize JPEG decompression object
      jpeg_decompress_struct cinfo;
      jpeg_error_mgr jerr;
      cinfo.err = jpeg_std_error(&jerr);
      jpeg_create_decompress(&cinfo);

      // Set up the custom memory source manager
      JMemorySourceManager srcManager(encoded_image_data, encoded_image_data_len);
      cinfo.src = &srcManager;

      // Read the JPEG header to get image info
      jpeg_read_header(&cinfo, TRUE);

      // Start decompression
      jpeg_start_decompress(&cinfo);

      // Allocate memory for the image
      std::vector<int64_t> output_dimensions{cinfo.output_height, cinfo.output_width, cinfo.output_components};
      uint8_t* imageBuffer = output.Allocate(output_dimensions);

      // Read the image data
      int row_stride = cinfo.output_width * cinfo.output_components;
      while (cinfo.output_scanline < cinfo.output_height) {
        uint8_t* row_ptr = imageBuffer + (cinfo.output_scanline * row_stride);
        jpeg_read_scanlines(&cinfo, &row_ptr, 1);
        if (srcManager.extError != kOrtxOK) {
          break;
        }
      }

      if (srcManager.extError != kOrtxOK) {
        return {kOrtxErrorInternal, "[ImageDecoder]: Failed to decode JPEG image."};
      }

      // Finish decompression
      jpeg_finish_decompress(&cinfo);
      jpeg_destroy_decompress(&cinfo);
    }
    return {};
  }

  class JMemorySourceManager : public jpeg_source_mgr {
   public:
    // Constructor
    JMemorySourceManager(const uint8_t* encoded_image_data, const int64_t encoded_image_data_len) {
      // Initialize source fields
      next_input_byte = reinterpret_cast<const JOCTET*>(encoded_image_data);
      bytes_in_buffer = static_cast<size_t>(encoded_image_data_len);
      init_source = &JMemorySourceManager::initSource;
      fill_input_buffer = &JMemorySourceManager::fillInputBuffer;
      skip_input_data = &JMemorySourceManager::skipInputData;
      resync_to_restart = jpeg_resync_to_restart;
      term_source = &JMemorySourceManager::termSource;
    }

    // Initialize source (no-op)
    static void initSource(j_decompress_ptr cinfo) {
      // No initialization needed
    }

    // Fill input buffer (not used here, always return FALSE)
    static boolean fillInputBuffer(j_decompress_ptr cinfo) {
      return FALSE;  // Buffer is managed manually
    }

    // Skip input data
    static void skipInputData(j_decompress_ptr cinfo, long num_bytes) {
      JMemorySourceManager* srcMgr = reinterpret_cast<JMemorySourceManager*>(cinfo->src);
      if (num_bytes > 0) {
        size_t bytes_to_skip = static_cast<size_t>(num_bytes);
        while (bytes_to_skip > srcMgr->bytes_in_buffer) {
          bytes_to_skip -= srcMgr->bytes_in_buffer;
          if (srcMgr->fillInputBuffer(cinfo)) {
            // Error: buffer ran out
            srcMgr->extError = kOrtxErrorCorruptData;
          }
        }
        srcMgr->next_input_byte += bytes_to_skip;
        srcMgr->bytes_in_buffer -= bytes_to_skip;
      }
    }

    // Terminate source (no-op)
    static void termSource(j_decompress_ptr cinfo) {
      // No cleanup needed
    }

    extError_t extError{kOrtxOK};  // Error handler
  };
};

}  // namespace ort_extensions::internal
