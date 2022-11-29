// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "encode_image.hpp"

// #include <opencv2/imgcodecs.hpp>
#include "png.h"
#include "jpeglib.h"

#include "vision/impl/png_encoder_decoder.hpp"

namespace ort_extensions {

namespace {
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
  // TODO: RGB is probably better if we're not using opencv
  const OrtValue* input_bgr = ort_.KernelContext_GetInput(context, 0ULL);
  const OrtTensorDimensions dimensions_bgr(ort_, input_bgr);

  if (dimensions_bgr.size() != 3 || dimensions_bgr[2] != 3) {
    // expect {H, W, C} as that's the inverse of what decode_image produces.
    // we have no way to check if it's BGR or RGB though
    ORT_CXX_API_THROW("[EncodeImage] requires rank 3 BGR input in channels last format.", ORT_INVALID_ARGUMENT);
  }

  if (extension_ == ".png") {
    PngEncoder encoder(ort_.GetTensorData<uint8_t>(input_bgr), dimensions_bgr);
    const auto& encoded_image = encoder.Encode();

    std::vector<int64_t> output_dimensions{static_cast<int64_t>(encoded_image.size())};
    OrtValue* output_value = ort_.KernelContext_GetOutput(context, 0,
                                                          output_dimensions.data(),
                                                          output_dimensions.size());

    uint8_t* data = ort_.GetTensorMutableData<uint8_t>(output_value);
    memcpy(data, encoded_image.data(), encoded_image.size());

  } else {
    uint8_t* missing_buffer = nullptr;
    unsigned long num_encoded_bytes = 0;
    EncodeJpg(missing_buffer, num_encoded_bytes);
  }
}
}  // namespace ort_extensions
