// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#pragma comment(lib, "Windowscodecs.lib")

#include <wincodec.h>
#include <wincodecsdk.h>

#include "op_def_struct.h"
#include "ext_status.h"


inline OrtxStatus image_decoder(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) {
  const auto& dimensions = input.Shape();
  if (dimensions.size() != 1ULL) {
    return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Only raw image formats are supported."};
  }

  // Get data & the length
  const uint8_t* encoded_image_data = input.Data();
  const int64_t encoded_image_data_len = input.NumberOfElement();

  // check it's a PNG image or JPEG image
  if (encoded_image_data_len < 8) {
    return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Invalid image data."};
  }

  OrtxStatus status{};

  CoInitialize(NULL);

  IWICImagingFactory* pIWICFactory = NULL;
  IWICBitmapDecoder* pIDecoder = NULL;
  IWICStream* pIWICStream = NULL;
  IWICBitmapFrameDecode* pIDecoderFrame = NULL;
  IWICComponentInfo* pIComponentInfo = NULL;
  WICPixelFormatGUID pixelFormat;

  // Create the COM imaging factory
  HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pIWICFactory));
  if (FAILED(hr)) {
    return {kOrtxErrorInternal, "[ImageDecoder]: Failed to create pIWICFactory."};
  }

  // Create a WIC stream to map onto the memory.
  hr = pIWICFactory->CreateStream(&pIWICStream);
  if (FAILED(hr)) {
    return {kOrtxErrorInternal, "[ImageDecoder]: Failed to create pIWICStream."};
  }

  static_assert(sizeof(uint8_t) == sizeof(unsigned char));

  // Initialize the stream with the memory pointer and size.
  hr = pIWICStream->InitializeFromMemory((unsigned char *)input.Data(), input.NumberOfElement());
  if (FAILED(hr)) {
    return {kOrtxErrorInternal, "[ImageDecoder]: Failed when pIWICStream->InitializeFromMemory."};
  }

  // Create a decoder for the stream.
  hr = pIWICFactory->CreateDecoderFromStream(pIWICStream,                     // Image to be decoded
                                             NULL,                            // Do not prefer a particular vendor
                                             WICDecodeMetadataCacheOnDemand,  // Cache metadata when needed
                                             &pIDecoder                       // Pointer to the decoder
  );
  if (FAILED(hr)) {
    return {kOrtxErrorInternal, "[ImageDecoder]: Failed to create pIDecoder."};
  }

  // Retrieve the first bitmap frame.
  hr = pIDecoder->GetFrame(0, &pIDecoderFrame);
  if (FAILED(hr)) {
    return {kOrtxErrorInternal, "[ImageDecoder]: Failed when pIDecoder->GetFrame."};
  }

  // Now get a POINTER to an instance of the Pixel Format
  hr = pIDecoderFrame->GetPixelFormat(&pixelFormat);
  if (FAILED(hr)) {
    return {kOrtxErrorInternal, "[ImageDecoder]: Failed when pIDecoderFrame->GetPixelFormat."};
  }

  hr = pIWICFactory->CreateComponentInfo(pixelFormat, &pIComponentInfo);
  if (FAILED(hr)) {
    return {kOrtxErrorInternal, "[ImageDecoder]: Failed when pIWICFactory->CreateComponentInfo."};
  }

  // Get IWICPixelFormatInfo from IWICComponentInfo
  IWICPixelFormatInfo2* pIPixelFormatInfo = NULL;
  hr = pIComponentInfo->QueryInterface(__uuidof(IWICPixelFormatInfo2), reinterpret_cast<void**>(&pIPixelFormatInfo));
  if (FAILED(hr)) {
    return {kOrtxErrorInternal, "[ImageDecoder]: Failed to query IWICPixelFormatInfo."};
  }

  UINT uiWidth = 0;
  UINT uiHeight = 0;

  hr = pIDecoderFrame->GetSize(&uiWidth, &uiHeight);
  if (FAILED(hr)) {
    return {kOrtxErrorInternal, "[ImageDecoder]: pIDecoderFrame->GetSize."};
  }

  const int height = static_cast<int>(uiHeight);
  const int width = static_cast<int>(uiWidth);
  const int channels = 3; // Asks for RGB

  std::vector<int64_t> output_dimensions{height, width, channels};
  uint8_t* decoded_image_data = output.Allocate(output_dimensions);
  if (decoded_image_data == nullptr) {
    return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Failed to allocate memory for decoded image data."};
  }

  IWICBitmapSource* pSource = pIDecoderFrame;

  if (pixelFormat != GUID_WICPixelFormat24bppRGB) {
    IWICBitmapSource* pConverted = NULL;
    hr = WICConvertBitmapSource(GUID_WICPixelFormat24bppRGB, pSource, &pConverted);
    if (FAILED(hr)) {
      return {kOrtxErrorInternal, "[ImageDecoder]: Failed when WICConvertBitmapSource."};
    }

    pSource->Release();
    pSource = pConverted;
  }

  int rowStride = uiWidth * sizeof(uint8_t) * channels;
  hr = pSource->CopyPixels(NULL, rowStride, output.SizeInBytes(), decoded_image_data);

  if (FAILED(hr)) {
    return {kOrtxErrorInternal, "[ImageDecoder]: Failed when pConvertedFrame->CopyPixels."};
  }

  pIComponentInfo->Release();
  pSource->Release();
  pIWICStream->Release();
  pIDecoder->Release();
  pIWICFactory->Release();

  CoUninitialize();

  return status;
}