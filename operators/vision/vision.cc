// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "decode_image.hpp"
#include "encode_image.hpp"

FxLoadCustomOpFactory LoadCustomOpClasses_Vision =
    LoadCustomOpClasses<CustomOpClassBegin,
                        ort_extensions::CustomOpDecodeImage,
                        ort_extensions::CustomOpEncodeImage>;
