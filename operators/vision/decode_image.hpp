// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ext_status.h"
#include "op_def_struct.h"

#if OCOS_ENABLE_VENDOR_IMAGE_CODECS
  #if WIN32
    #include "image_decoder_win32.hpp"
  #elif __APPLE__
    #include "image_decoder_darwin.hpp"
  #else
    #include "image_decoder.hpp"
  #endif
#else
  #include "image_decoder.hpp"
#endif

namespace ort_extensions {
struct DecodeImage: public internal::DecodeImage {
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {return {};}
  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) const{
    return internal::DecodeImage::Compute(input, output);
  }
};

}  // namespace ort_extensions
