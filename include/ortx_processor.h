// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// C ABI header file for the onnxruntime-extensions tokenization module

#pragma once

#include "ortx_utils.h"

// typedefs to create/dispose function flood, and to make the API more C++ friendly with less casting
typedef OrtxObject OrtxProcessor;
typedef OrtxObject OrtxRawImages;
typedef OrtxObject OrtxImageProcessorResult;

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Create a processor object with the specified processor definition
 *
 * \param processor Pointer to store the created processor object
 * \param processor_def The processor definition, either a path to the processor directory or a JSON string, and is
 * utf-8 encoded. \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxCreateProcessor(OrtxProcessor** processor, const char* processor_def);

/** \brief Dispose of the processor object
 *
 * \param processor The processor object to dispose
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxLoadImages(const char** image_paths, size_t num_images, OrtxRawImages** images,
                                        size_t* num_images_loaded);

/**
 * @brief Preprocesses the given raw images using the specified processor.
 *
 * This function applies preprocessing operations on the raw images using the provided processor.
 * The result of the preprocessing is stored in the `OrtxImageProcessorResult` object.
 *
 * @param processor A pointer to the `OrtxProcessor` object used for preprocessing.
 * @param images A pointer to the `OrtxRawImages` object containing the raw images to be processed.
 * @param result A pointer to the `OrtxImageProcessorResult` object to store the preprocessing result.
 * @return An `extError_t` value indicating the success or failure of the preprocessing operation.
 */
extError_t ORTX_API_CALL OrtxImagePreProcess(OrtxProcessor* processor, OrtxRawImages* images,
                                             OrtxImageProcessorResult** result);

/** \brief Clear the outputs of the processor
 *
 * \param processor The processor object
 * \param result The result object to clear
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxClearOutputs(OrtxProcessor* processor, OrtxImageProcessorResult* result);

#ifdef __cplusplus
}
#endif
