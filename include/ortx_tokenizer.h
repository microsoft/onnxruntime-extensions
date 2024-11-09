// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// C ABI header file for the onnxruntime-extensions tokenization module

#pragma once

#include "ortx_utils.h"

// typedefs to create/dispose function flood, and to make the API more C++ friendly with less casting
typedef OrtxObject OrtxTokenizer;
typedef OrtxObject OrtxStringArray;
typedef OrtxObject OrtxTokenId2DArray;
typedef OrtxObject OrtxDetokenizerCache;

struct OrtxTokenizerBlob {
  const char* config_json_blob;
  const char* vocab_json_blob;
  const char* token_module_blob;
  const char* raw_model_blob;
  const char* reserved_blob_1;

  const size_t config_blob_len;
  const size_t vocab_blob_len;
  const size_t token_module_blob_len;
  const size_t raw_model_blob_len;
  const size_t reserved_blob_1_len;

#ifdef __cplusplus
  OrtxTokenizerBlob(const std::string_view& config_json_blob,
                    const std::string_view& vocab_json_blob,
                    const std::string_view& token_module_blob,
                    const std::string_view& raw_model_blob)
      : config_json_blob(config_json_blob.data()), vocab_json_blob(vocab_json_blob.data()),
        token_module_blob(token_module_blob.data()), raw_model_blob(raw_model_blob.data()),
        config_blob_len(config_json_blob.size()),
        vocab_blob_len(vocab_json_blob.size()), token_module_blob_len(token_module_blob.size()),
        raw_model_blob_len(raw_model_blob.size()), reserved_blob_1(nullptr),
        reserved_blob_1_len(0) {}
#endif
};


#ifdef __cplusplus
extern "C" {
#endif

/** \brief Create a tokenizer object with the specified tokenizer path
 *
 * \param tokenizer Pointer to store the created tokenizer object
 * \param tokenizer_path The path to the tokenizer directory, which is utf-8 encoded
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxCreateTokenizer(OrtxTokenizer** tokenizer, const char* tokenizer_path);

/** \brief Create a tokenizer object with the specified tokenizer blob
 *
 * \param tokenizer Pointer to store the created tokenizer object
 * \param tokenizer_blob Pointer to the tokenizer blob
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxCreateTokenizerFromBlob(OrtxTokenizer** tokenizer, const struct OrtxTokenizerBlob* tokenizer_blob);


/** \brief Tokenize the input using the specified tokenizer
 *
 * \param tokenizer Pointer to the tokenizer object
 * \param input Array of input strings
 * \param batch_size Number of input strings in the batch
 * \param output Pointer to store the tokenized result
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxTokenize(
    const OrtxTokenizer* tokenizer, const char* input[], size_t batch_size, OrtxTokenId2DArray** output);


/**
 * Converts a token to its corresponding ID.
 *
 * @param tokenizer The tokenizer object.
 * @param input The input token to be converted.
 * @param output Pointer to store the converted token ID.
 * @return The error code indicating the success or failure of the conversion.
 */
extError_t ORTX_API_CALL OrtxConvertTokenToId(const OrtxTokenizer* tokenizer, const char* token, extTokenId_t* id);

/**
 * @brief Retrieves the decoder prompt IDs from the tokenizer.
 *
 * This function retrieves the decoder prompt IDs from the specified tokenizer.
 *
 * @param tokenizer A pointer to the OrtxTokenizer object.
 * @param batch_size The size of the batch.
 * @param lang The language for the Whisper model decoding, like 'en'. Can be NULL, which is no id in the output.
 * @param task The task for the model, like 'translation' or 'transcribe'. Can be NULL, which is no id in the output.
 * @param no_timestamps Flag indicating whether to include timestamps in the output. 1 is true, 0 is false.
 * @param output A pointer to the OrtxTokenId2DArray object to store the output.
 * @return An extError_t value indicating the success or failure of the operation.
 */
extError_t ORTX_API_CALL OrtxGetDecoderPromptIds(
    const OrtxTokenizer* tokenizer, size_t batch_size, const char* lang, const char* task, int no_timestamps, OrtxTokenId2DArray** output);

/** \brief Detokenize the input using the specified tokenizer
 *
 * \param tokenizer Pointer to the tokenizer object
 * \param input Pointer to the input token IDs
 * \param output Pointer to store the detokenized result
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxDetokenize(
    const OrtxTokenizer* tokenizer, const OrtxTokenId2DArray* input, OrtxStringArray** output);

/** \brief Detokenize the input using the specified tokenizer (1D version)
 *
 * \param tokenizer Pointer to the tokenizer object
 * \param input Pointer to the input token IDs
 * \param len Length of the input token IDs array
 * \param output Pointer to store the detokenized result
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxDetokenize1D(
    const OrtxTokenizer* tokenizer, const extTokenId_t* input, size_t len, OrtxStringArray** output);

/** \brief Detokenize the input using the specified tokenizer with caching
 *
 * \param tokenizer Pointer to the tokenizer object
 * \param cache Pointer to the detokenizer cache
 * \param next_id Next token ID to detokenize
 * \param text_out Pointer to store the detokenized text
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxDetokenizeCached(
    const OrtxTokenizer* tokenizer, OrtxDetokenizerCache* cache, extTokenId_t next_id, const char** text_out);

/** \brief Get the length of the string array
 *
 * \param string_array Pointer to the string array
 * \param length Pointer to store the length of the string array
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxStringArrayGetBatch(const OrtxStringArray* string_array, size_t* length);

/** \brief Get the item at the specified index from the string array
 *
 * \param string_array Pointer to the string array
 * \param index Index of the item to retrieve
 * \param item Pointer to store the retrieved item
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxStringArrayGetItem(const OrtxStringArray* string_array, size_t index, const char** item);

/** \brief Get the batch size of the token ID 2D array
 *
 * \param token_id_2d_array Pointer to the token ID 2D array
 * \param length Pointer to store the batch size
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxTokenId2DArrayGetBatch(const OrtxTokenId2DArray* token_id_2d_array, size_t* length);

/** \brief Get the item at the specified index from the token ID 2D array
 *
 * \param token_id_2d_array Pointer to the token ID 2D array
 * \param index Index of the item to retrieve
 * \param item Pointer to store the retrieved item
 * \param length Pointer to store the length of the item
 * \return Error code indicating the success or failure of the operation
 */
extError_t ORTX_API_CALL OrtxTokenId2DArrayGetItem(
    const OrtxTokenId2DArray* token_id_2d_array, size_t index, const extTokenId_t** item, size_t* length);

#ifdef __cplusplus
}
#endif
