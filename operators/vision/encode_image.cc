// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "encode_image.hpp"

// #include <opencv2/imgcodecs.hpp>
#include "png.h"
#include "jpeglib.h"

namespace ort_extensions {

namespace {
void EncodePng() {
  int x, y;

  int width, height;
  png_byte color_type;
  png_byte bit_depth;

  png_structp png_ptr;
  png_infop info_ptr;
  int number_of_passes;
  png_bytep* row_pointers;

  /* initialize stuff */
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  info_ptr = png_create_info_struct(png_ptr);

  png_init_io(png_ptr, 0);
  png_set_IHDR(png_ptr, info_ptr, width, height,
               bit_depth, color_type, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
  png_write_info(png_ptr, info_ptr);
  png_write_image(png_ptr, row_pointers);
  png_write_end(png_ptr, NULL);
}

void EncodeJpg(uint8_t*& buffer, unsigned long& num_bytes) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer[1]; /* pointer to JSAMPLE row[s] */
  int row_stride;          /* physical row width in image buffer */

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  jpeg_mem_dest(&cinfo, &buffer, &num_bytes);

  cinfo.image_width = 600; /* image width and height, in pixels */
  cinfo.image_height = 600;
  cinfo.input_components = 3;     /* # of color components per pixel */
  cinfo.in_color_space = JCS_RGB; /* colorspace of input image */
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, /*quality*/ 90, TRUE /* limit to baseline-JPEG values */);

  jpeg_start_compress(&cinfo, TRUE);

  row_stride = cinfo.image_width * 3; /* JSAMPLEs per row in image_buffer */
  JSAMPLE* image_buffer = nullptr;    // allocated externally

  while (cinfo.next_scanline < cinfo.image_height) {
    /* jpeg_write_scanlines expects an array of pointers to scanlines.
     * Here the array is only one element long, but you could pass
     * more than one scanline at a time if that's more convenient.
     */
    row_pointer[0] = &image_buffer[cinfo.next_scanline * row_stride];
    (void)jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
}
}  // namespace

void KernelEncodeImage ::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_bgr = ort_.KernelContext_GetInput(context, 0ULL);
  const OrtTensorDimensions dimensions_bgr(ort_, input_bgr);

  if (dimensions_bgr.size() != 3 || dimensions_bgr[2] != 3) {
    // expect {H, W, C} as that's the inverse of what decode_image produces.
    // we have no way to check if it's BGR or RGB though
    ORT_CXX_API_THROW("[EncodeImage] requires rank 3 BGR input in channels last format.", ORT_INVALID_ARGUMENT);
  }

  // Get data & the length
  std::vector<int32_t> height_x_width{static_cast<int32_t>(dimensions_bgr[0]),   // H
                                      static_cast<int32_t>(dimensions_bgr[1])};  // W

  //// data is const uint8_t but opencv2 wants void*.
  // const void* bgr_data = ort_.GetTensorData<uint8_t>(input_bgr);
  // const cv::Mat bgr_image(height_x_width, CV_8UC3, const_cast<void*>(bgr_data));

  //// don't know output size ahead of time so need to encode and then copy to output
  std::vector<uint8_t> encoded_image;
  // if (!cv::imencode(extension_, bgr_image, encoded_image)) {
  //   ORT_CXX_API_THROW("[EncodeImage] Image encoding failed.", ORT_INVALID_ARGUMENT);
  // }

  // Setup output & copy to destination
  std::vector<int64_t> output_dimensions{static_cast<int64_t>(1234)};  // encoded_image.size())};
  OrtValue* output_value = ort_.KernelContext_GetOutput(context, 0,
                                                        output_dimensions.data(),
                                                        output_dimensions.size());

  uint8_t* data = ort_.GetTensorMutableData<uint8_t>(output_value);
  memcpy(data, encoded_image.data(), encoded_image.size());
}
}  // namespace ort_extensions
