// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "png.h"
#include "jpeglib.h"
#include "op_def_struct.h"
#include "ext_status.h"

#include "encode_image.hpp"


namespace ort_extensions {

void KernelEncodeImage::Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) const {
  const auto& dimensions_bgr = input.Shape();
  if (dimensions_bgr.size() != 3 || dimensions_bgr[2] != 3) {
    ORTX_CXX_API_THROW("[EncodeImage] requires rank 3 BGR input in channels last format.", ORT_INVALID_ARGUMENT);
  }

  std::vector<int32_t> height_x_width{static_cast<int32_t>(dimensions_bgr[0]),   // H
                                      static_cast<int32_t>(dimensions_bgr[1])};  // W
  const int color_space = 3;
  const uint8_t* bgr_data = input.Data();
  unsigned char* outbuffer = nullptr;
  std::vector<uint8_t> png_buffer;
  size_t outsize = 0;

  auto rgb_data = std::make_unique<uint8_t[]>(height_x_width[0] * height_x_width[1] * color_space);
  for (int y = 0; y < height_x_width[0]; ++y) {
    for (int x = 0; x < height_x_width[1]; ++x) {
      rgb_data[(y * height_x_width[1] + x) * color_space + 0] = bgr_data[(y * height_x_width[1] + x) * color_space + 2];
      rgb_data[(y * height_x_width[1] + x) * color_space + 1] = bgr_data[(y * height_x_width[1] + x) * color_space + 1];
      rgb_data[(y * height_x_width[1] + x) * color_space + 2] = bgr_data[(y * height_x_width[1] + x) * color_space + 0];
    }
  }

  if (extension_ == ".jpg") {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_mem_dest(&cinfo, &outbuffer, &outsize);

    cinfo.image_width = height_x_width[1];
    cinfo.image_height = height_x_width[0];
    cinfo.input_components = color_space;
    cinfo.in_color_space = JCS_RGB;

    // compression parameters is compatible with opencv
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 95, TRUE);
    cinfo.optimize_coding = FALSE;
    cinfo.restart_interval = 0;
    cinfo.q_scale_factor[0] = jpeg_quality_scaling(-1);
    cinfo.q_scale_factor[1] = jpeg_quality_scaling(-1);

    const int sampling_factor = 0x221111; // 4:2:0  IMWRITE_JPEG_SAMPLING_FACTOR_420
    cinfo.comp_info[0].v_samp_factor = (sampling_factor >> 16 ) & 0xF;
    cinfo.comp_info[0].h_samp_factor = (sampling_factor >> 20 ) & 0xF;
    cinfo.comp_info[1].v_samp_factor = 1;
    cinfo.comp_info[1].h_samp_factor = 1;
    // jpeg_default_qtables( &cinfo, TRUE );

    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
      row_pointer[0] = (JSAMPROW)&rgb_data[cinfo.next_scanline * cinfo.image_width * color_space];
      jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
  } else if (extension_ == ".png") {
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

    png_set_write_fn(png_ptr, &png_buffer, [](png_structp png_ptr, png_bytep data, png_size_t length) {
      auto p = reinterpret_cast<std::vector<uint8_t>*>(png_get_io_ptr(png_ptr));
      p->insert(p->end(), data, data + length);
    }, nullptr);

    // sync with openCV parameters
    png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
    png_set_compression_level(png_ptr, 1);
    png_set_compression_strategy(png_ptr, 3);

    png_set_IHDR(png_ptr, info_ptr, height_x_width[1], height_x_width[0], 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    for (int y = 0; y < height_x_width[0]; ++y) {
      png_write_row(png_ptr, (png_bytep)&rgb_data[y * height_x_width[1] * color_space]);
    }

    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);

    outbuffer = png_buffer.data();
    outsize = png_buffer.size();
  } else {
    ORTX_CXX_API_THROW("[EncodeImage] Unsupported image format.", ORT_INVALID_ARGUMENT);
  }

  std::vector<int64_t> output_dimensions{static_cast<int64_t>(outsize)};
  uint8_t* data = output.Allocate(output_dimensions);
  memcpy(data, outbuffer, outsize);

  if (outbuffer != png_buffer.data() && outbuffer != nullptr) {
    free(outbuffer);
  }
}

}  // namespace ort_extensions
