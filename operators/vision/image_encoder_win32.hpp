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

  bool JpgSupportsBgr() const{ return true; }

  OrtxStatus EncodeJpg(const uint8_t* source_data, bool source_is_bgr, int32_t width, int32_t height,
                uint8_t** outbuffer, size_t* outsize) const {
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

    return EncodeWith(GUID_ContainerFormatJpeg, options, values, source_data, source_is_bgr,
        width, height, outbuffer,outsize);
  }

  bool pngSupportsBgr() const{ return true; }

  OrtxStatus EncodePng(const uint8_t* source_data, bool source_is_bgr, int32_t width, int32_t height,
                uint8_t** outbuffer,size_t* outsize) const{
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

    return EncodeWith(GUID_ContainerFormatPng, options, values, source_data, source_is_bgr, width, height,
                      outbuffer, outsize);
  }

  ~EncodeImage() {
    if (pIWICFactory_) {
      pIWICFactory_->Release();
      pIWICFactory_ = NULL;
    }
    CoUninitialize();
  }

  private:
   OrtxStatus EncodeWith(GUID conatinerFormatGUID, std::vector<PROPBAG2> options, std::vector<VARIANT> values,
                   const uint8_t* source_data, bool source_is_bgr, int32_t width, int32_t height, uint8_t** outbuffer,
                   size_t* outsize) const {
     const uint8_t* encode_source_data = source_data;
     winrt::com_ptr<IStream> pOutputStream;
     winrt::com_ptr<IWICBitmapEncoder> pIEncoder;
     winrt::com_ptr<IWICBitmapFrameEncode> pBitmapFrame;
     winrt::com_ptr<IPropertyBag2> pPropertyBag;

     HRESULT hr = ::CreateStreamOnHGlobal(NULL, FALSE, pOutputStream.put());
     if (FAILED(hr)) {
       return errorWithHr_("Failed when CreateStreamOnHGlobal.", hr);
     }

     hr = pIWICFactory_->CreateEncoder(conatinerFormatGUID, NULL, pIEncoder.put());
     if (FAILED(hr)) {
       return errorWithHr_("Failed when CreateEncoder.", hr);
     }

     hr = pIEncoder->Initialize(pOutputStream.get(), WICBitmapEncoderNoCache);

     if (FAILED(hr)) {
       return errorWithHr_("Failed when pIEncoder->Initialize.", hr);
     }

     hr = pIEncoder->CreateNewFrame(pBitmapFrame.put(), pPropertyBag.put());

     if (FAILED(hr)) {
       return errorWithHr_("Failed when pIEncoder->CreateNewFrame.", hr);
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
       return errorWithHr_("Failed when pBitmapFrame->Initialize.", hr);
     }

     hr = pBitmapFrame->SetSize(width, height);
     if (FAILED(hr)) {
       return errorWithHr_("Failed when pBitmapFrame->SetSize.", hr);
     }

     WICPixelFormatGUID pixelFormatGUID = GUID_WICPixelFormat24bppRGB;

     hr = pBitmapFrame->SetPixelFormat(&pixelFormatGUID);
     if (FAILED(hr)) {
       return errorWithHr_("Failed when pBitmapFrame->SetPixelFormat.", hr);
     }

     if (!IsEqualGUID(pixelFormatGUID, GUID_WICPixelFormat24bppRGB) &&
         IsEqualGUID(pixelFormatGUID, GUID_WICPixelFormat24bppBGR)) {
       // 24bppRGB not supported natively by encoder. Encoder expects BGR.
       // This is true for both PNG & JPEG encoder.
       assert(source_is_bgr);
       encode_source_data = source_data;
     } else {
       // TODO: Handle this if we need to support more formats.
     }

     UINT cbStride = (width * 24 + 7) / 8 /***WICGetStride***/;
     UINT cbBufferSize = height * cbStride;

     hr = pBitmapFrame->WritePixels(height, cbStride, cbBufferSize, (BYTE*)encode_source_data);

     if (FAILED(hr)) {
       return errorWithHr_("pBitmapFrame->WritePixels.", hr);
     }

     hr = pBitmapFrame->Commit();
     if (FAILED(hr)) {
       return errorWithHr_("pBitmapFrame->Commit.", hr);
     }
     hr = pIEncoder->Commit();
     if (FAILED(hr)) {
       return errorWithHr_("pIEncoder->Commit.", hr);
     }

     STATSTG outputStreamStat;
     hr = pOutputStream->Stat(&outputStreamStat, 0);
     if (FAILED(hr)) {
       return errorWithHr_("pOutputStream->Stat.", hr);
     }

     const size_t size = static_cast<size_t>(outputStreamStat.cbSize.QuadPart);
     void *buffer = malloc(size);
     if (buffer == NULL) {
       return errorWithHr_("malloc failed.", hr);
     }
     ULONG cbRead = 0;
     // Seek to beginning for Read() to work properly.
     hr = pOutputStream->Seek(LARGE_INTEGER(), STREAM_SEEK_SET, NULL);
     if (FAILED(hr)) {
       return errorWithHr_("pOutputStream->Seek.", hr);
     }
     hr = pOutputStream->Read(buffer, static_cast<ULONG>(size), &cbRead);
     if (FAILED(hr)) {
       return errorWithHr_("pOutputStream->Read.", hr);
     }
     *outbuffer = (uint8_t*)buffer;
     *outsize = size;
     return {};
   }

   OrtxStatus errorWithHr_(const std::string message, HRESULT hr) const {
     return {kOrtxErrorInternal, "[ImageEncoder]: " + message + " HRESULT: " + std::to_string(hr)};
   }

   IWICImagingFactory* pIWICFactory_{NULL};
};
}
