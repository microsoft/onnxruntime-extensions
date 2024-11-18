// C ABI header file for the onnxruntime-extensions tokenization module

#pragma once

#include "ortx_utils.h"

typedef OrtxObject OrtxFeatureExtractor;
typedef OrtxObject OrtxRawAudios;

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

/**
 * Loads a collection of audio files into memory.
 *
 * This function loads a collection of audio files specified by the `audio_paths` array
 * into memory and returns a pointer to the loaded audio data in the `audios` parameter.
 *
 * @param audios A pointer to a pointer that will be updated with the loaded audio data.
 *               The caller is responsible for freeing the memory allocated for the audio data.
 * @param audio_paths An array of strings representing the paths to the audio files to be loaded.
 * @param num_audios The number of audio files to be loaded.
 * 
 * @return An `extError_t` value indicating the success or failure of the operation.
 */
extError_t ORTX_API_CALL OrtxLoadAudios(OrtxRawAudios** audios, const char* const* audio_paths, size_t num_audios);

/**
 * @brief Creates an array of raw audio objects, which refers to the audio data and sizes provided.
 *
 * This function creates an array of raw audio objects based on the provided data and sizes. The data will be stored in
 * the `audios` parameter.
 *
 * @param audios Pointer to the variable that will hold the created raw audio objects.
 * @param data Array of pointers to the audio data.
 * @param sizes Array of pointers to the sizes of the audio data.
 * @param num_audios Number of audio objects to create.
 *
 * @return extError_t Error code indicating the success or failure of the operation.
 */
extError_t ORTX_API_CALL OrtxCreateRawAudios(OrtxRawAudios** audios, const void* data[], const int64_t sizes[], size_t num_audios);

/**
 * @brief Calculates the log mel spectrogram for a given audio using the specified feature extractor.
 *
 * This function takes an instance of the OrtxFeatureExtractor struct, an instance of the OrtxRawAudios struct,
 * and a pointer to an OrtxTensorResult pointer. It calculates the log mel spectrogram for the given audio using
 * the specified feature extractor and stores the result in the provided log_mel pointer.
 *
 * @param extractor The feature extractor to use for calculating the log mel spectrogram.
 * @param audio The raw audio data to process.
 * @param log_mel A pointer to an OrtxTensorResult pointer where the result will be stored.
 * @return An extError_t value indicating the success or failure of the operation.
 */
extError_t ORTX_API_CALL OrtxSpeechLogMel(OrtxFeatureExtractor* extractor, OrtxRawAudios* audio, OrtxTensorResult** log_mel);

#ifdef __cplusplus
}
#endif
