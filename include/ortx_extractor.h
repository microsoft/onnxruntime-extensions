// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// C ABI header file for the onnxruntime-extensions tokenization module

#pragma once

#include "ortx_utils.h"

typedef OrtxObject OrtxFeatureExtractor;
typedef OrtxObject OrtxRawAudios;
typedef OrtxObject OrtxTensorResult;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a feature extractor object.
 *
 * This function creates a feature extractor object based on the provided feature definition.
 *
 * @param[out] extractor Pointer to a pointer to the created feature extractor object.
 * @param[in] fe_def The feature definition used to create the feature extractor.
 *
 * @return An error code indicating the result of the operation.
 */
extError_t ORTX_API_CALL OrtxCreateSpeechFeatureExtractor(OrtxFeatureExtractor** extractor, const char* fe_def);

extError_t ORTX_API_CALL OrtxLoadAudios(OrtxRawAudios** audios, const char* const* audio_paths, size_t num_audios);

extError_t ORTX_API_CALL OrtxSpeechLogMel(OrtxFeatureExtractor* extractor, OrtxRawAudios* audio, OrtxTensorResult** log_mel);

#ifdef __cplusplus
}
#endif
