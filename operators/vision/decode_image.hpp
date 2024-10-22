// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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
