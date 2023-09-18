// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "onnxruntime_cxx_api.h"

using LibraryHandle = std::unique_ptr<void, void(*)(void*)>;

// Register ORT Extensions custom ops with `session_options`.
// Returns a unique_ptr to the ORT Extensions shared library handle that will manage freeing it, if applicable.
LibraryHandle RegisterExtOps(Ort::SessionOptions& session_options);
