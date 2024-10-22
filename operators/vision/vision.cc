// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "decode_image.hpp"
#include "encode_image.hpp"
#include "draw_bounding_box.hpp"

const std::vector<const OrtCustomOp*>& VisionLoader() {
  static OrtOpLoader op_loader(CustomCpuStruct("EncodeImage", ort_extensions::KernelEncodeImage),
                               CustomCpuStructV2("DecodeImage", DecodeImage),
                               CustomCpuStruct("DrawBoundingBoxes", ort_extensions::DrawBoundingBoxes));
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Vision = VisionLoader;