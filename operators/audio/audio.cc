// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ocos.h"
#ifdef ENABLE_DR_LIBS
#include "audio_decoder.hpp"
#endif  // ENABLE_DR_LIBS

FxLoadCustomOpFactory LoadCustomOpClasses_Audio = []()-> CustomOpArray& {
  static OrtOpLoader op_loader(
    []() { return nullptr; }
#ifdef ENABLE_DR_LIBS
    ,
    CustomCpuStruct("AudioDecoder", AudioDecoder)
#endif
  );

  return op_loader.GetCustomOps();
};
