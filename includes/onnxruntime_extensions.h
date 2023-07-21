// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifdef SPM_BUILD
#include "onnxruntime/onnxruntime_c_api.h"
#else
#include "onnxruntime_c_api.h"
#endif

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
