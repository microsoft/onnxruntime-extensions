// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <CoreFoundation/CoreFoundation.h>
#include <CoreServices/CoreServices.h>
#include <ImageIO/ImageIO.h>

#include "op_def_struct.h"
#include "ext_status.h"

namespace ort_extensions::internal {

struct EncodeImage {
  OrtxStatus OnInit() {
    float compression = 0.95;

    CFStringRef optionKeys[2];
    CFTypeRef optionValues[2];
    optionKeys[0] = kCGImagePropertyHasAlpha;
    optionValues[0] = (CFTypeRef)kCFBooleanFalse;
    optionKeys[1] = kCGImageDestinationLossyCompressionQuality;
    optionValues[1] = CFNumberCreate(NULL, kCFNumberFloatType, &compression);

    imageDestinationOptions_ = CFDictionaryCreate(NULL, (const void**)optionKeys, (const void**)optionValues, 2,
                                             &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    return {};
  }

  bool JpgSupportsBgr() const{ return false; }

  void EncodeJpg(const uint8_t* source_data, bool source_is_bgr, int32_t width, int32_t height,
                uint8_t** outbuffer, size_t* outsize) const{
    return EncodeWith(kUTTypeJPEG, imageDestinationOptions_, source_data, source_is_bgr,
                      width, height, outbuffer, outsize);
  }

  bool pngSupportsBgr() const{ return false; }

  void EncodePng(const uint8_t* source_data, bool source_is_bgr, int32_t width, int32_t height,
                uint8_t** outbuffer,size_t* outsize) const{
    return EncodeWith(kUTTypePNG, imageDestinationOptions_, source_data, source_is_bgr,
                      width, height, outbuffer, outsize);
  }

  ~EncodeImage() { CFRelease(imageDestinationOptions_); }

private:
  void EncodeWith(CFStringRef type, CFDictionaryRef option, const uint8_t* source_data, bool source_is_bgr,
                 int32_t width, int32_t height, uint8_t** outbuffer, size_t* outsize) const{
    const uint8_t* raw_pixel_source_data = source_data;

    size_t cbStride = (width * 24 + 7) / 8;
    size_t cbBufferSize = height * cbStride;
    CFDataRef rawImageData = CFDataCreate(NULL, raw_pixel_source_data, cbBufferSize);
    if (rawImageData == nullptr) {
      ORTX_CXX_API_THROW("[ImageDecoder]: Failed to create CFData.", ORT_RUNTIME_EXCEPTION);
    }
    CGDataProviderRef provider = CGDataProviderCreateWithCFData(rawImageData);
    CGColorSpaceRef colorspace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGImageRef image = CGImageCreate(width, // width
                                     height, // height
                                     8, // bitsPerComponent
                                     24, // bitsPerPixel
                                     3 * width, // bytesPerRow
                                     colorspace,
                                     kCGImageAlphaNone, // bitmapInfo
                                     provider,
                                     NULL, // decode
                                     true, // shouldInterpolate
                                     kCGRenderingIntentDefault // intent
                                     );
    CFRelease(colorspace);
    CFRelease(provider);
    CFRelease(rawImageData);
    if (image == nullptr) {
      ORTX_CXX_API_THROW("[ImageEncoder]: Failed to CGImageCreate.", ORT_RUNTIME_EXCEPTION);
    }
    CFMutableDataRef result = CFDataCreateMutable(NULL, 0);
    if (result == nullptr) {
      ORTX_CXX_API_THROW("[ImageEncoder]: Failed to CFDataCreateMutable.", ORT_RUNTIME_EXCEPTION);
    }
    CGImageDestinationRef imageDest = CGImageDestinationCreateWithData(result, type, 1 /* count */, NULL);
    if (imageDest == nullptr) {
      ORTX_CXX_API_THROW("[ImageEncoder]: Failed to CGImageDestinationCreateWithData.", ORT_RUNTIME_EXCEPTION);
    }
    CGImageDestinationAddImage(imageDest, image, option);
    CGImageDestinationFinalize(imageDest);
    CFRelease(imageDest);
    CFRelease(image);

    int size = CFDataGetLength(result);
    *outbuffer = (uint8_t*)malloc(size);
    CFDataGetBytes(result, CFRangeMake(0, size), *outbuffer);
    CFRelease(result);
    *outsize = size;
  }

  CFDictionaryRef imageDestinationOptions_{NULL};
};

}
