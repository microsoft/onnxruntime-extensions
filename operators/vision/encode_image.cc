// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "png.h"
#include "jpeglib.h"
#include "op_def_struct.h"
#include "ext_status.h"

#include "encode_image.hpp"


namespace ort_extensions {

void KernelEncodeImage::Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) const {
  const auto dimensions_bgr = input.Shape();

  if (dimensions_bgr.size() != 3 || dimensions_bgr[2] != 3) {
    ORTX_CXX_API_THROW("[EncodeImage] requires rank 3 BGR input in channels last format.", ORT_INVALID_ARGUMENT);
  }

  std::vector<int32_t> height_x_width{static_cast<int32_t>(dimensions_bgr[0]), static_cast<int32_t>(dimensions_bgr[1])};
  const uint8_t* bgr_data = input.Data();
  unsigned char* outbuffer = nullptr;
  std::vector<uint8_t> buffer;
  size_t outsize = 0;

  if (extension_ == ".jpg") {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_mem_dest(&cinfo, &outbuffer, &outsize);

    cinfo.image_width = height_x_width[1];
    cinfo.image_height = height_x_width[0];
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
      row_pointer[0] = (JSAMPROW)&bgr_data[cinfo.next_scanline * cinfo.image_width * 3];
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

    png_set_write_fn(png_ptr, &buffer, [](png_structp png_ptr, png_bytep data, png_size_t length) {
      auto p = reinterpret_cast<std::vector<uint8_t>*>(png_get_io_ptr(png_ptr));
      p->insert(p->end(), data, data + length);
    }, nullptr);

    png_set_IHDR(png_ptr, info_ptr, height_x_width[1], height_x_width[0], 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    for (int y = 0; y < height_x_width[0]; ++y) {
      png_write_row(png_ptr, (png_bytep)&bgr_data[y * height_x_width[1] * 3]);
    }

    png_write_end(png_ptr, info_ptr);
    png_destroy_write_struct(&png_ptr, &info_ptr);

    outbuffer = buffer.data();
    outsize = buffer.size();
  } else {
    ORTX_CXX_API_THROW("[EncodeImage] Unsupported image format.", ORT_INVALID_ARGUMENT);
  }

  std::vector<int64_t> output_dimensions{static_cast<int64_t>(outsize)};
  uint8_t* data = output.Allocate(output_dimensions);
  memcpy(data, outbuffer, outsize);

  if (outbuffer != buffer.data() && outbuffer != nullptr) {
    free(outbuffer);
  }
}

}  // namespace ort_extensions
