// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ortx_processor.h"
#include "ext_status.h"
#include <vector>

namespace ort_extensions {
OrtxStatus ValidateVideoOptions(const OrtxVideoLoadOptions& options, const OrtxVideoLimits& limits);
OrtxStatus DecodeVideo(const char* path, const OrtxVideoLoadOptions& options, const OrtxVideoLimits& limits,
                       std::vector<std::vector<uint8_t>>& pixels, std::vector<OrtxFrame>& frames,
                       std::vector<OrtxVideoSample>& samples, OrtxVideoMetadata& metadata);
}  // namespace ort_extensions
