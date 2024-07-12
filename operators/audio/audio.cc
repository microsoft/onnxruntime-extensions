// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ocos.h"
#ifdef ENABLE_DR_LIBS
#include "audio_decoder.h"
#endif  // ENABLE_DR_LIBS

FxLoadCustomOpFactory LoadCustomOpClasses_Audio = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(
#ifdef ENABLE_DR_LIBS
      CustomCpuStructV2("AudioDecoder", AudioDecoder),
#endif
      []() { return nullptr; });

  return op_loader.GetCustomOps();
};
