// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#pragma comment(lib, "Windowscodecs.lib")

#include <wincodec.h>
#include <wincodecsdk.h>
#include <winrt/base.h>

#include "op_def_struct.h"
#include "ext_status.h"

namespace ort_extensions::internal {
struct EncodeImage {
  OrtxStatus OnInit() {
    HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED | COINIT_DISABLE_OLE1DDE);
    if (FAILED(hr)) {
      return errorWithHr_("Failed when CoInitialize.", hr);
    }
    // Create the COM imaging factory
    hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pIWICFactory_));
    if (FAILED(hr)) {
      return errorWithHr_("Failed to create pIWICFactory.", hr);
    }

    return {};
  }

  void EncodeJpgFromRgb(const uint8_t* rgb_data, int32_t width, int32_t height, uint8_t** outbuffer,
                        size_t* outsize) const {
    std::vector<PROPBAG2> options;
    std::vector<VARIANT> values;

    {
      PROPBAG2 option = {0};
      option.pstrName = L"ImageQuality";
      options.push_back(option);
      VARIANT varValue;
      VariantInit(&varValue);
      varValue.vt = VT_R4;
      varValue.fltVal = static_cast<FLOAT>(0.95);
      values.push_back(varValue);
    }

    {
      PROPBAG2 option = {0};
      option.pstrName = L"JpegYCrCbSubsampling";
      options.push_back(option);
      VARIANT varValue;
      VariantInit(&varValue);
      varValue.vt = VT_UI1;
      varValue.bVal = WICJpegYCrCbSubsampling420;
      values.push_back(varValue);
    }

    return EncodeWith(GUID_ContainerFormatJpeg, options, values, rgb_data, width, height, outbuffer, outsize);
  }

  void EncodePngFromRgb(const uint8_t* rgb_data, int32_t width, int32_t height, uint8_t** outbuffer,
                        size_t* outsize) const{
    std::vector<PROPBAG2> options;
    std::vector<VARIANT> values;

    {
      PROPBAG2 option = {0};
      option.pstrName = L"InterlaceOption";
      options.push_back(option);
      VARIANT varValue;
      VariantInit(&varValue);
      varValue.vt = VT_BOOL;
      varValue.boolVal = FALSE;
      values.push_back(varValue);
    }

    {
      PROPBAG2 option = {0};
      option.pstrName = L"FilterOption";
      options.push_back(option);
      VARIANT varValue;
      VariantInit(&varValue);
      varValue.vt = VT_UI1;
      varValue.bVal = WICPngFilterSub;
      values.push_back(varValue);
    }

    return EncodeWith(GUID_ContainerFormatPng, options, values, rgb_data, width, height, outbuffer, outsize);
  }

  ~EncodeImage() {
    pIWICFactory_.detach();
    CoUninitialize();
  }

  private:
   void EncodeWith(GUID conatinerFormatGUID, std::vector<PROPBAG2> options, std::vector<VARIANT> values,
       const uint8_t* rgb_data, int32_t width, int32_t height, uint8_t** outbuffer, size_t* outsize) const{
     const uint8_t* source_data = rgb_data;
     winrt::com_ptr<IStream> pOutputStream;
     winrt::com_ptr<IWICBitmapEncoder> pIEncoder;
     winrt::com_ptr<IWICBitmapFrameEncode> pBitmapFrame;
     winrt::com_ptr<IPropertyBag2> pPropertyBag;

     HRESULT hr = ::CreateStreamOnHGlobal(NULL, FALSE, pOutputStream.put());
     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("Failed when CreateStreamOnHGlobal.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }

     hr = pIWICFactory_->CreateEncoder(conatinerFormatGUID, NULL, pIEncoder.put());
     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("Failed when CreateEncoder.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }

     hr = pIEncoder->Initialize(pOutputStream.get(), WICBitmapEncoderNoCache);

     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("Failed when pIEncoder->Initialize.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }

     hr = pIEncoder->CreateNewFrame(pBitmapFrame.put(), pPropertyBag.put());

     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("Failed when pIEncoder->CreateNewFrame.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }

     assert(options.size() == values.size());

     int idx = 0;
     for (auto option : options) {
       auto varValue = values[idx];
       hr = pPropertyBag->Write(1, &option, &varValue);
       idx++;
     }

     hr = pBitmapFrame->Initialize(pPropertyBag.get());
     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("Failed when pBitmapFrame->Initialize.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }

     hr = pBitmapFrame->SetSize(width, height);
     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("Failed when pBitmapFrame->SetSize.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }

     WICPixelFormatGUID pixelFormatGUID = GUID_WICPixelFormat24bppRGB;

     hr = pBitmapFrame->SetPixelFormat(&pixelFormatGUID);
     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("Failed when pBitmapFrame->SetPixelFormat.", hr).Message(),
                          ORT_RUNTIME_EXCEPTION);
     }

     if (!IsEqualGUID(pixelFormatGUID, GUID_WICPixelFormat24bppRGB) &&
         IsEqualGUID(pixelFormatGUID, GUID_WICPixelFormat24bppBGR)) {
       // 24bppRGB not supported by encoder. Convert source data and use 24bppBGR.
       const int color_space = 3;
       auto bgr_data = (uint8_t*)malloc(height * width * color_space);
       for (int32_t y = 0; y < height; ++y) {
         for (int32_t x = 0; x < width; ++x) {
           bgr_data[(y * width + x) * color_space + 0] = rgb_data[(y * width + x) * color_space + 2];
           bgr_data[(y * width + x) * color_space + 1] = rgb_data[(y * width + x) * color_space + 1];
           bgr_data[(y * width + x) * color_space + 2] = rgb_data[(y * width + x) * color_space + 0];
         }
       }

       source_data = (uint8_t*)bgr_data;
     }

     UINT cbStride = (width * 24 + 7) / 8 /***WICGetStride***/;
     UINT cbBufferSize = height * cbStride;

     hr = pBitmapFrame->WritePixels(height, cbStride, cbBufferSize, (BYTE*)source_data);

     if (source_data != rgb_data) {
       free((void*)source_data);
     }

     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("pBitmapFrame->WritePixels.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }

     hr = pBitmapFrame->Commit();
     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("pBitmapFrame->Commit.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }
     hr = pIEncoder->Commit();
     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("pIEncoder->Commit.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }

     STATSTG outputStreamStat;
     hr = pOutputStream->Stat(&outputStreamStat, 0);
     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("pOutputStream->Stat.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }

     const size_t size = static_cast<size_t>(outputStreamStat.cbSize.QuadPart);
     void *buffer = malloc(size);
     if (buffer == NULL) {
       ORTX_CXX_API_THROW(errorWithHr_("malloc failed.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }
     ULONG cbRead = 0;
     // Seek to beginning for Read() to work properly.
     hr = pOutputStream->Seek(LARGE_INTEGER(), STREAM_SEEK_SET, NULL);
     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("pOutputStream->Seek.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }
     hr = pOutputStream->Read(buffer, static_cast<ULONG>(size), &cbRead);
     if (FAILED(hr)) {
       ORTX_CXX_API_THROW(errorWithHr_("pOutputStream->Read.", hr).Message(), ORT_RUNTIME_EXCEPTION);
     }
     *outbuffer = (uint8_t*)buffer;
     *outsize = size;
   }

   OrtxStatus errorWithHr_(const std::string message, HRESULT hr) const {
     return {kOrtxErrorInternal, "[ImageEncoder]: " + message + " HRESULT: " + std::to_string(hr)};
   }
   winrt::com_ptr<IWICImagingFactory> pIWICFactory_;
};
}
