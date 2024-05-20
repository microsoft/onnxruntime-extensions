// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "image_processor.h"

using namespace ort_extensions;

extError_t OrtxCreateProcessor(OrtxProcessor** processor, const char* def) {
  if (processor == nullptr || def == nullptr) {
    return kOrtxErrorInvalidArgument;
  }

  auto proc_ptr = std::make_unique<ImageProcessor>();
  ReturnableStatus status = proc_ptr->Init(def);
  if (status.IsOk()) {
    *processor = static_cast<OrtxProcessor*>(proc_ptr.release());
  } else {
    *processor = nullptr;
  }

  return status.Code();
}
