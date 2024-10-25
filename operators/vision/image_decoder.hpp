// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "png.h"
#include "jpeglib.h"
#include "op_def_struct.h"
#include "ext_status.h"

namespace ort_extensions::internal {

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

struct DecodeImage {
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
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

    OrtxStatus status{};
    if (png_sig_cmp(encoded_image_data, 0, 8) == 0) {
      // Decode the PNG image
      png_image image;
      std::memset(&image, 0, sizeof(image));  // Use std::memset for clarity
      image.version = PNG_IMAGE_VERSION;

      if (png_image_begin_read_from_memory(&image, encoded_image_data, static_cast<size_t>(encoded_image_data_len)) ==
          0) {
        return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Failed to read PNG image."};
      }

      image.format = PNG_FORMAT_RGB;  // Ensure you have the appropriate format
      const int height = image.height;
      const int width = image.width;
      const int channels = PNG_IMAGE_PIXEL_CHANNELS(image.format);  // Calculates the number of channels based on format

      std::vector<int64_t> output_dimensions{height, width, channels};

      uint8_t* decoded_image_data = output.Allocate(output_dimensions);
      if (decoded_image_data == nullptr) {
        return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Failed to allocate memory for decoded image data."};
      }

      if (png_image_finish_read(&image, nullptr, decoded_image_data, 0, nullptr) == 0) {
        return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Failed to decode PNG image."};
      }
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
        status = {srcManager.extError, "[ImageDecoder]: Failed to decode JPEG image."};
      }

      // Finish decompression
      jpeg_finish_decompress(&cinfo);
      jpeg_destroy_decompress(&cinfo);
    }

    return status;
    }
};

}  // namespace ort_extensions::internal
