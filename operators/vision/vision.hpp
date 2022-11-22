// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(ENABLE_VISION)
#include "ocos.h"
#include "decode_image.hpp"
#include "encode_image.hpp"

FxLoadCustomOpFactory LoadCustomOpClasses_PPP_Vision =
    LoadCustomOpClasses<CustomOpClassBegin,
                        ort_extensions::CustomOpDecodeImage,
                        ort_extensions::CustomOpEncodeImage>;
#endif
