// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <CoreFoundation/CoreFoundation.h>
#include <ImageIO/ImageIO.h>

#include "op_def_struct.h"
#include "ext_status.h"

namespace ort_extensions::internal {

struct EncodeImage {
  OrtxStatus OnInit() {}

};

}