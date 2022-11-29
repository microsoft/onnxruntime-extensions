#include "png_encoder_decoder.hpp"

/****************************************************************************************\
    This part of the file implements PNG codec on base of libpng library,
    in particular, this code is based on example.c from libpng
    (see otherlibs/_graphics/readme.txt for copyright notice)
    and png2bmp sample from libpng distribution (Copyright (C) 1999-2001 MIYASAKA Masaru)
\****************************************************************************************/

#include "png.h"
#include "zlib.h"

#include <cassert>

#if defined _MSC_VER && _MSC_VER >= 1200
// interaction between '_setjmp' and C++ object destruction is non-portable
#pragma warning(disable : 4611)
#endif

namespace ort_extensions {

bool PngDecoder::IsPng(const uint8_t* bytes, uint64_t num_bytes) {
    // '\0x89PGN\r\n<sub>\n'
  constexpr const uint8_t signature[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};  // check start of file for this
  constexpr int signature_len = 8;

  return num_bytes > signature_len && memcmp(bytes, signature, signature_len) == 0;
}

PngDecoder::~PngDecoder() {
  if (png_ptr_) {
    png_structp png_ptr = (png_structp)png_ptr_;
    png_infop info_ptr = (png_infop)info_ptr_;
    png_infop end_info = (png_infop)end_info_;
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
  }
}

// read a chunk from bytes_
void PngDecoder::ReadDataFromBuf(void* _png_ptr, uint8_t* dst, size_t size) {
  png_structp png_ptr = (png_structp)_png_ptr;
  PngDecoder* decoder = (PngDecoder*)(png_get_io_ptr(png_ptr));
  assert(decoder);

  if (decoder->cur_offset_ + size > decoder->NumBytes()) {
    png_error(png_ptr, "PNG input buffer is incomplete");
    return;
  }

  memcpy(dst, decoder->Bytes() + decoder->cur_offset_, size);
  decoder->cur_offset_ += size;
}

bool PngDecoder::ReadHeader() {
  bool result = false;

  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
  assert(png_ptr);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  png_infop end_info = png_create_info_struct(png_ptr);

  assert(info_ptr);
  assert(end_info);

  png_ptr_ = png_ptr;
  info_ptr_ = info_ptr;
  end_info_ = end_info;

  if (setjmp(png_jmpbuf(png_ptr)) == 0) {
    png_set_read_fn(png_ptr, this, (png_rw_ptr)ReadDataFromBuf);

    png_uint_32 wdth, hght;
    int bit_depth, color_type, num_trans = 0;
    png_bytep trans;
    png_color_16p trans_values;

    png_read_info(png_ptr, info_ptr);
    png_get_IHDR(png_ptr, info_ptr, &wdth, &hght,
                 &bit_depth, &color_type, 0, 0, 0);

    // set the shape based on what Decode will do. doesn't necessarily match the original image though as we're
    // going to throw away any alpha channel and drop 16 bit depth to 8.
    SetShape(static_cast<int>(hght), static_cast<int>(wdth), 3);

    color_type_ = color_type;
    bit_depth_ = bit_depth;

    // TODO: Do we need to handle these combinations?
    //       We want to end up with 3 channel RGB though so they might not be relevant given we (I think) set the png
    //       code to throw away the alpha channel, reduce 16-bit to 8, and convert grayscale to 3 channel.
    //
    //
    // if (bit_depth <= 8 || bit_depth == 16) {
    //  switch (color_type) {
    //    case PNG_COLOR_TYPE_RGB:
    //    case PNG_COLOR_TYPE_PALETTE:
    //      png_get_tRNS(png_ptr, info_ptr, &trans, &num_trans, &trans_values);
    //      if (num_trans > 0)
    //        m_type = CV_8UC4;
    //      else
    //        m_type = CV_8UC3;
    //      break;
    //    case PNG_COLOR_TYPE_GRAY_ALPHA:
    //    case PNG_COLOR_TYPE_RGB_ALPHA:
    //      m_type = CV_8UC4;
    //      break;
    //    default:
    //      m_type = CV_8UC1;
    //  }
    //  if (bit_depth == 16)
    //    m_type = CV_MAKETYPE(CV_16U, CV_MAT_CN(m_type));
    //
    //}
    result = true;
  }

  return result;
}

bool PngDecoder::DecodeImpl(uint8_t* output, uint64_t out_bytes) {
  bool result = false;

  png_structp png_ptr = (png_structp)png_ptr_;
  png_infop info_ptr = (png_infop)info_ptr_;
  png_infop end_info = (png_infop)end_info_;

  if (setjmp(png_jmpbuf(png_ptr)) == 0) {
    if (bit_depth_ == 16) {
      png_set_strip_16(png_ptr);
    }

    png_set_strip_alpha(png_ptr);

    if (color_type_ == PNG_COLOR_TYPE_PALETTE) {
      png_set_palette_to_rgb(png_ptr);
    }

    if ((color_type_ & PNG_COLOR_MASK_COLOR) == 0 && bit_depth_ < 8) {
      png_set_expand_gray_1_2_4_to_8(png_ptr);
    }

    // NOTE: NOT calling png_set_bgr so that output is RGB instead of the slightly weird BGR.
    if (!(color_type_ & PNG_COLOR_MASK_COLOR)) {
      png_set_gray_to_rgb(png_ptr);  // Gray->RGB
    }

    png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    const auto& shape = Shape();
    auto height = shape[0];
    auto width = shape[1];
    auto channels = shape[2];

    std::vector<uint8_t*> row_pointers(height, nullptr);
    auto row_size = png_get_rowbytes(png_ptr, info_ptr);
    assert(row_size == width * channels);  // check assumption

    for (int row = 0; row < height; ++row) {
      row_pointers[row] = output + (row * row_size);
    }

    png_read_image(png_ptr, row_pointers.data());
    png_read_end(png_ptr, end_info);

    result = true;
  }

  return result;
}

/////////////////////// PngEncoder ///////////////////

// PngEncoder::PngEncoder() {
//   m_description = "Portable Network Graphics files (*.png)";
//   m_buf_supported = true;
// }
//
// PngEncoder::~PngEncoder() {
// }
//
// bool PngEncoder::isFormatSupported(int depth) const {
//   return depth == CV_8U || depth == CV_16U;
// }
//
// ImageEncoder PngEncoder::newEncoder() const {
//   return makePtr<PngEncoder>();
// }
//
// void PngEncoder::writeDataToBuf(void* _png_ptr, uchar* src, size_t size) {
//   if (size == 0)
//     return;
//   png_structp png_ptr = (png_structp)_png_ptr;
//   PngEncoder* encoder = (PngEncoder*)(png_get_io_ptr(png_ptr));
//   assert(encoder && encoder->m_buf);
//   size_t cursz = encoder->m_buf->size();
//   encoder->m_buf->resize(cursz + size);
//   memcpy(&(*encoder->m_buf)[cursz], src, size);
// }
//
// void PngEncoder::flushBuf(void*) {
// }
//
// bool PngEncoder::write(const Mat& img, const std::vector<int>& params) {
//   png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
//   png_infop info_ptr = 0;
//   FILE* volatile f = 0;
//   int y, width = img.cols, height = img.rows;
//   int depth = img.depth(), channels = img.channels();
//   volatile bool result = false;
//   AutoBuffer<uchar*> buffer;
//
//   if (depth != CV_8U && depth != CV_16U)
//     return false;
//
//   if (png_ptr) {
//     info_ptr = png_create_info_struct(png_ptr);
//
//     if (info_ptr) {
//       if (setjmp(png_jmpbuf(png_ptr)) == 0) {
//         if (m_buf) {
//           png_set_write_fn(png_ptr, this,
//                            (png_rw_ptr)writeDataToBuf, (png_flush_ptr)flushBuf);
//         } else {
//           f = fopen(m_filename.c_str(), "wb");
//           if (f)
//             png_init_io(png_ptr, (png_FILE_p)f);
//         }
//
//         int compression_level = -1;                           // Invalid value to allow setting 0-9 as valid
//         int compression_strategy = IMWRITE_PNG_STRATEGY_RLE;  // Default strategy
//         bool isBilevel = false;
//
//         for (size_t i = 0; i < params.size(); i += 2) {
//           if (params[i] == IMWRITE_PNG_COMPRESSION) {
//             compression_strategy = IMWRITE_PNG_STRATEGY_DEFAULT;  // Default strategy
//             compression_level = params[i + 1];
//             compression_level = MIN(MAX(compression_level, 0), Z_BEST_COMPRESSION);
//           }
//           if (params[i] == IMWRITE_PNG_STRATEGY) {
//             compression_strategy = params[i + 1];
//             compression_strategy = MIN(MAX(compression_strategy, 0), Z_FIXED);
//           }
//           if (params[i] == IMWRITE_PNG_BILEVEL) {
//             isBilevel = params[i + 1] != 0;
//           }
//         }
//
//         if (m_buf || f) {
//           if (compression_level >= 0) {
//             png_set_compression_level(png_ptr, compression_level);
//           } else {
//             // tune parameters for speed
//             // (see http://wiki.linuxquestions.org/wiki/Libpng)
//             png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
//             png_set_compression_level(png_ptr, Z_BEST_SPEED);
//           }
//           png_set_compression_strategy(png_ptr, compression_strategy);
//
//           png_set_IHDR(png_ptr, info_ptr, width, height, depth == CV_8U ? isBilevel ? 1 : 8 : 16,
//                        channels == 1 ? PNG_COLOR_TYPE_GRAY : channels == 3 ? PNG_COLOR_TYPE_RGB
//                                                                            : PNG_COLOR_TYPE_RGBA,
//                        PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
//                        PNG_FILTER_TYPE_DEFAULT);
//
//           png_write_info(png_ptr, info_ptr);
//
//           if (isBilevel)
//             png_set_packing(png_ptr);
//
//           png_set_bgr(png_ptr);
//           if (!isBigEndian())
//             png_set_swap(png_ptr);
//
//           buffer.allocate(height);
//           for (y = 0; y < height; y++)
//             buffer[y] = img.data + y * img.step;
//
//           png_write_image(png_ptr, buffer.data());
//           png_write_end(png_ptr, info_ptr);
//
//           result = true;
//         }
//       }
//     }
//   }
//
//   png_destroy_write_struct(&png_ptr, &info_ptr);
//   if (f) fclose((FILE*)f);
//
//   return result;
// }

}  // namespace ort_extensions
