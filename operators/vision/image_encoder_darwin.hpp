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

  OrtxStatus EncodeJpg(const uint8_t* source_data, bool source_is_bgr, int32_t width, int32_t height,
                uint8_t** outbuffer, size_t* outsize) const{
    return EncodeWith(kUTTypeJPEG, imageDestinationOptions_, source_data, source_is_bgr,
                      width, height, outbuffer, outsize);
  }

  bool pngSupportsBgr() const{ return false; }

  OrtxStatus EncodePng(const uint8_t* source_data, bool source_is_bgr, int32_t width, int32_t height,
                uint8_t** outbuffer,size_t* outsize) const{
    return EncodeWith(kUTTypePNG, imageDestinationOptions_, source_data, source_is_bgr,
                      width, height, outbuffer, outsize);
  }

  ~EncodeImage() { CFRelease(imageDestinationOptions_); }

private:
  OrtxStatus EncodeWith(CFStringRef type, CFDictionaryRef option, const uint8_t* source_data, bool source_is_bgr,
                 int32_t width, int32_t height, uint8_t** outbuffer, size_t* outsize) const{
    if (source_data == nullptr || outbuffer == nullptr || outsize == nullptr || !IsValidEncodeImageSize(width, height)) {
      return {kOrtxErrorInvalidArgument, "[EncodeImage] Invalid image arguments or dimensions."};
    }
    assert(!source_is_bgr);
    *outbuffer = nullptr;
    *outsize = 0;

    const uint8_t* raw_pixel_source_data = source_data;

    const size_t cbStride = (static_cast<size_t>(width) * 24 + 7) / 8;
    const size_t cbBufferSize = static_cast<size_t>(height) * cbStride;
    CFDataRef rawImageData = CFDataCreate(NULL, raw_pixel_source_data, static_cast<CFIndex>(cbBufferSize));
    if (rawImageData == nullptr) {
      return {kOrtxErrorInternal, "[ImageDecoder]: Failed to create CFData."};
    }
    CGDataProviderRef provider = CGDataProviderCreateWithCFData(rawImageData);
    CGColorSpaceRef colorspace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGImageRef image = CGImageCreate(static_cast<size_t>(width), // width
                                     static_cast<size_t>(height), // height
                                     8, // bitsPerComponent
                                     24, // bitsPerPixel
                                     cbStride, // bytesPerRow
                                     colorspace,
                                     kCGImageAlphaNone, // bitmapInfo
                                     provider,
                                     NULL, // decode
                                     true, // shouldInterpolate
                                     kCGRenderingIntentDefault // intent
                                     );
    if (colorspace != nullptr) {
      CFRelease(colorspace);
    }
    if (provider != nullptr) {
      CFRelease(provider);
    }
    CFRelease(rawImageData);
    if (image == nullptr) {
      return {kOrtxErrorInternal, "[ImageEncoder]: Failed to CGImageCreate."};
    }
    CFMutableDataRef result = CFDataCreateMutable(NULL, 0);
    if (result == nullptr) {
      CFRelease(image);
      return {kOrtxErrorInternal, "[ImageEncoder]: Failed to CFDataCreateMutable."};
    }
    CGImageDestinationRef imageDest = CGImageDestinationCreateWithData(result, type, 1 /* count */, NULL);
    if (imageDest == nullptr) {
      CFRelease(result);
      CFRelease(image);
      return {kOrtxErrorInternal, "[ImageEncoder]: Failed to CGImageDestinationCreateWithData."};
    }
    CGImageDestinationAddImage(imageDest, image, option);
    const bool finalized = CGImageDestinationFinalize(imageDest);
    CFRelease(imageDest);
    CFRelease(image);
    if (!finalized) {
      CFRelease(result);
      return {kOrtxErrorInternal, "[ImageEncoder]: Failed to finalize CGImageDestination."};
    }

    const CFIndex size = CFDataGetLength(result);
    if (size > 0) {
      *outbuffer = static_cast<uint8_t*>(malloc(static_cast<size_t>(size)));
      if (*outbuffer == nullptr) {
        CFRelease(result);
        return {kOrtxErrorOutOfMemory, "[ImageEncoder]: Failed to allocate output buffer."};
      }
      CFDataGetBytes(result, CFRangeMake(0, size), *outbuffer);
    }
    CFRelease(result);
    *outsize = static_cast<size_t>(size);
    return {};
  }

  CFDictionaryRef imageDestinationOptions_{NULL};
};

}
