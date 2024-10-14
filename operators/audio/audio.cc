// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ocos.h"

#ifdef OCOS_ENABLE_VENDOR_AUDIO_CODECS
  #if defined(__APPLE__)
  #include "audio_decoder_darwin.h"
  #endif
#elif defined(ENABLE_DR_LIBS)
  #include "audio_decoder_drlib.h"
#endif  // ENABLE_DR_LIBS

FxLoadCustomOpFactory LoadCustomOpClasses_Audio = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(
#if defined(ENABLE_DR_LIBS) || defined(OCOS_ENABLE_VENDOR_AUDIO_CODECS)
      CustomCpuStructV2("AudioDecoder", AudioDecoder),
#endif
      []() { return nullptr; });

  return op_loader.GetCustomOps();
};
