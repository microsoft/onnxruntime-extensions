// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ocos.h"
#include "asr_diarization_merge.h"
#ifdef ENABLE_DR_LIBS
#include "audio_decoder.h"
#endif  // ENABLE_DR_LIBS

FxLoadCustomOpFactory LoadCustomOpClasses_Audio = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(
      CustomCpuStructV2("AsrDiarizationMergeRaw", ort_extensions::AsrDiarizationMergeRaw),
      CustomCpuStructV2("AsrDiarizationMergeSegments", ort_extensions::AsrDiarizationMergeSegments),
#ifdef ENABLE_DR_LIBS
      CustomCpuStructV2("AudioDecoder", AudioDecoder),
#endif
      []() { return nullptr; });

  return op_loader.GetCustomOps();
};
