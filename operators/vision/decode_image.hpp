// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

#include <cstdint>

namespace ort_extensions {
void decode_image(const ortc::TensorT<uint8_t>& input,
                  ortc::TensorT<uint8_t>& output);
}  // namespace ort_extensions
