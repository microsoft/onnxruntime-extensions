// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "zlib.h"
#if ZLIB_VERNUM != 0x12b0
// the following is a trick to show the invalid version number for the diagnosis.
#define STR_VERSION(x) STR_NUM(x)
#define STR_NUM(x) #x
#pragma message "Invalid zlib version:  " STR_VERSION(ZLIB_VERNUM)
#error "stopped"
#endif

#include "png.h"
#if WIN32
// Fix redefinition in jmorecfg.h
#include <basetsd.h>
#endif
#include "jpeglib.h"
#include "op_def_struct.h"
#include "ext_status.h"

#include "encode_image.hpp"

namespace ort_extensions::internal {
struct EncodeImage {
  OrtxStatus OnInit() { return {}; }

  bool JpgSupportsBgr() const{ return false; }
  void EncodeJpg(const uint8_t* rgb_data, bool source_is_bgr, int32_t width, int32_t height, uint8_t** outbuffer,
                size_t* outsize) const {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_mem_dest(&cinfo, outbuffer, outsize);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    // compression parameters is compatible with opencv
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 95, TRUE);
    cinfo.optimize_coding = FALSE;
    cinfo.restart_interval = 0;
    cinfo.q_scale_factor[0] = jpeg_quality_scaling(-1);
    cinfo.q_scale_factor[1] = jpeg_quality_scaling(-1);

    const int32_t sampling_factor = 0x221111;  // 4:2:0  IMWRITE_JPEG_SAMPLING_FACTOR_420
    cinfo.comp_info[0].v_samp_factor = (sampling_factor >> 16) & 0xF;
    cinfo.comp_info[0].h_samp_factor = (sampling_factor >> 20) & 0xF;
    cinfo.comp_info[1].v_samp_factor = 1;
    cinfo.comp_info[1].h_samp_factor = 1;
    // jpeg_default_qtables( &cinfo, TRUE );

    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
      row_pointer[0] = (JSAMPROW)&rgb_data[cinfo.next_scanline * cinfo.image_width * 3];
      jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
  }

  bool pngSupportsBgr() const{ return false; }

  void EncodePng(const uint8_t* rgb_data, bool source_is_bgr, int32_t width, int32_t height,
                uint8_t** outbuffer, size_t* outsize) const {
    std::vector<uint8_t> png_buffer;
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
      ORTX_CXX_API_THROW("[EncodeImage] PNG create write struct failed.", ORT_INVALID_ARGUMENT);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
      png_destroy_write_struct(&png_ptr, nullptr);
      ORTX_CXX_API_THROW("[EncodeImage] PNG create info struct failed.", ORT_INVALID_ARGUMENT);
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
      png_destroy_write_struct(&png_ptr, &info_ptr);
      ORTX_CXX_API_THROW("[EncodeImage] PNG encoding failed.", ORT_INVALID_ARGUMENT);
    }

    png_set_write_fn(
        png_ptr, &png_buffer,
        [](png_structp png_ptr, png_bytep data, png_size_t length) {
          auto p = reinterpret_cast<std::vector<uint8_t>*>(png_get_io_ptr(png_ptr));
          p->insert(p->end(), data, data + length);
        },
        nullptr);

    // sync with openCV parameters
    png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
    png_set_compression_level(png_ptr, 1);
    png_set_compression_strategy(png_ptr, 3);

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    for (int32_t y = 0; y < height; ++y) {
      png_write_row(png_ptr, (png_bytep)&rgb_data[y * width * 3]);
    }

    png_write_flush(png_ptr);
    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);

    const auto size = png_buffer.size();
    *outbuffer = (uint8_t*)malloc(size);
    std::copy(png_buffer.data(), png_buffer.data() + size, *outbuffer);
    *outsize = size;
  }
};
}  // namespace ort_extensions::internal
