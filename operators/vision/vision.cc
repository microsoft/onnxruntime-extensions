// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "decode_image.hpp"
#include "encode_image.hpp"

const std::vector<const OrtCustomOp*>& VisionLoader() {
  using namespace ort_extensions;
  static OrtOpLoader op_loader(LiteCustomOp("DecodeImage", decode_image),
                               LiteCustomOpStruct("EncodeImage", KernelEncodeImage));
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Vision = VisionLoader;
