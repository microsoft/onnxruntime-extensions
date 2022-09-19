// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_c_api.h"

#ifdef _WIN32
#define ORTX_EXPORT __declspec(dllexport)
#else
#define ORTX_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

ORTX_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);

#ifdef __cplusplus
}
#endif
