// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// C ABI header file for the onnxruntime-extensions tokenization module

#pragma once

#include "ortx_utils.h"

// typedefs to create/dispose function flood, and to make the API more C++ friendly with less casting
typedef OrtxObject OrtxProcessor;

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Create a processor object with the specified processor definition
 *
 * \param processor Pointer to store the created processor object
 * \param processor_def The processor definition
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxCreateProcessor(OrtxProcessor** processor, const char* processor_def);

#ifdef __cplusplus
}
#endif
