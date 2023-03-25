// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "decode_image.hpp"
#include "encode_image.hpp"

const std::vector<const OrtCustomOp*>& VisionLoader() {
  static OrtOpLoader op_loader(LiteCustomOp("DecodeImage", ort_extensions::decode_image),
                               BuildCustomOp(ort_extensions::CustomOpEncodeImage));
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Vision = VisionLoader;
