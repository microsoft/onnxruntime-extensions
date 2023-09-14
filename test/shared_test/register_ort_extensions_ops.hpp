// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "onnxruntime_cxx_api.h"

using LibraryHandle = std::unique_ptr<void, void(*)(void*)>;

LibraryHandle RegisterOrtExtensionsOps(Ort::SessionOptions& session_options);
