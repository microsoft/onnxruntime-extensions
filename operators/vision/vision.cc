// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "decode_image.hpp"
#include "encode_image.hpp"
#include "draw_bounding_box.hpp"

const std::vector<const OrtCustomOp*>& VisionLoader() {
  static OrtOpLoader op_loader(CustomCpuStructV2("DecodeImage", ort_extensions::DecodeImage),
                               CustomCpuStruct("EncodeImage", ort_extensions::KernelEncodeImage),
                               CustomCpuStruct("DrawBoundingBoxes", ort_extensions::DrawBoundingBoxes));
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Vision = VisionLoader;
