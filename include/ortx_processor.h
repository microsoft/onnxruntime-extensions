// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// C ABI header file for the onnxruntime-extensions tokenization module

#pragma once

#include "ortx_utils.h"

// typedefs to create/dispose function flood, and to make the API more C++ friendly with less casting
typedef OrtxObject OrtxProcessor;
typedef OrtxObject OrtxRawImages;
typedef OrtxObject OrtxRawFrames;

/** An interleaved RGB8 frame. Row padding is allowed; channels/pixels have strides 1/3.
 * data_size covers the readable span through the final row, not just packed pixel bytes.
 * timestamps are finite nonnegative seconds on the original presentation timeline.
 */
typedef struct {
  const void* data;
  size_t data_size;
  int64_t width;
  int64_t height;
  size_t row_stride;
  double timestamp_seconds;
} OrtxFrame;

typedef struct {
  size_t struct_size;
  size_t max_frames;
  size_t max_bytes;
  int64_t max_dimension;
} OrtxFrameLimits;

typedef struct {
  size_t struct_size;
  int64_t height;
  int64_t width;
} OrtxFramePreProcessOptions;

typedef struct {
  size_t struct_size;
  int64_t source_height;
  int64_t source_width;
  int64_t height;
  int64_t width;
} OrtxImagePreProcessOptions;

typedef enum { kOrtxVideoSamplingFps = 0, kOrtxVideoSamplingCount = 1, kOrtxVideoSamplingTimes = 2 } OrtxVideoSampling;

typedef struct {
  size_t struct_size;
  size_t max_frames;
  size_t max_rgb_bytes;
  size_t max_file_bytes;
  size_t max_read_bytes;
  size_t max_packets;
  size_t max_decoded_frames;
  size_t max_targets;
  int64_t max_dimension;
} OrtxVideoLimits;

typedef struct {
  size_t struct_size;
  OrtxVideoSampling sampling;
  double sampling_fps;
  size_t frame_count;
  const double* target_times;
  size_t target_count;
  double start_seconds;
  double end_seconds;
  int has_end;
} OrtxVideoLoadOptions;

typedef struct {
  size_t struct_size;
  int64_t time_base_num;
  int64_t time_base_den;
  int64_t origin_pts;
  size_t requested_samples;
  size_t selected_frames;
} OrtxVideoMetadata;

typedef struct {
  double target_seconds;
  size_t frame_index;
  int64_t source_pts;
  size_t source_frame_index;
} OrtxVideoSample;

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

/** Create an owned snapshot of one ordered frame sequence.
 * Validate all descriptors and aggregate quotas before copying any pixels.
 * Input descriptors/buffers must remain readable and immutable during this synchronous call.
 * Equal timestamps are allowed; decreasing timestamps are rejected, never sorted.
 * Null limits selects 64 frames, 256 MiB packed RGB and a 16384-pixel dimension limit.
 * Dispose the result through OrtxDispose. Failure sets *frames to null.
 */
extError_t ORTX_API_CALL OrtxCreateRawFrames(OrtxRawFrames** frames, const OrtxFrame* descriptors, size_t num_frames,
                                             const OrtxFrameLimits* limits);

/** Validate one sequence without allocating/copying pixels; return its packed RGB byte count.
 * Uses the same descriptor and quota rules as OrtxCreateRawFrames.
 */
extError_t ORTX_API_CALL OrtxValidateRawFrames(const OrtxFrame* descriptors, size_t num_frames,
                                               const OrtxFrameLimits* limits, size_t* packed_bytes);

extError_t ORTX_API_CALL OrtxGetRawFrameCount(const OrtxRawFrames* frames, size_t* count);

/** Retrieve a borrowed, tightly packed RGB8 view, valid until the frames handle is disposed. */
extError_t ORTX_API_CALL OrtxGetRawFrame(const OrtxRawFrames* frames, size_t index, OrtxFrame* descriptor);

/** Load a finite local MP4/H.264 video into the same owned RGB frame representation.
 * Exactly one sampling policy is required; fields belonging to other policies must be zero/null.
 * Times are elapsed presentation times from the original video start, even for a selected range.
 * Select the first presentation-order frame at/after each target within [start,end).
 * FPS/count deduplicate frames and expose underfill; count requires an explicit finite range.
 * Explicit times must be strictly increasing and select distinct frames, otherwise loading fails.
 * Progressive square-pixel SDR input only; orthogonal rotation/reflection is applied once.
 * Sound is ignored. Decode errors and work/quota exhaustion never return partial success.
 * Null limits uses 64 frames, 256 MiB RGB, 512 MiB file, 1 GiB read work,
 * 100000 packets, 4096 decoded frames/targets and maximum dimension 4096.
 * A cooperative 30-second deadline and 16-megapixel allocation guard also apply.
 * Codec workspace is separately bounded by decoder geometry, not the RGB output quota.
 * Decoder-disabled builds return kOrtxErrorNotImplemented. Failure sets *frames to null.
 */
extError_t ORTX_API_CALL OrtxLoadVideo(OrtxRawFrames** frames, const char* path, const OrtxVideoLoadOptions* options,
                                       const OrtxVideoLimits* limits);

/** Validate sampling/quota metadata without opening a file or allocating/copying pixels.
 * Uses exactly the loader's rules. Decoder-disabled builds return kOrtxErrorNotImplemented.
 */
extError_t ORTX_API_CALL OrtxValidateVideoLoadOptions(const OrtxVideoLoadOptions* options,
                                                      const OrtxVideoLimits* limits);

/** Metadata is available only for file-loaded frames; initialize metadata->struct_size. */
extError_t ORTX_API_CALL OrtxGetVideoMetadata(const OrtxRawFrames* frames, OrtxVideoMetadata* metadata);

/** One record per requested target. Uncovered FPS/count targets have frame_index=SIZE_MAX.
 * Frame indices address OrtxGetRawFrame; source frame indices are original decode-order indices.
 */
extError_t ORTX_API_CALL OrtxGetVideoSample(const OrtxRawFrames* frames, size_t index, OrtxVideoSample* sample);

/** Preprocess one RGB frame sequence using a pipeline starting in Resize/Rescale and ending in PatchImage.
 * Resize/normalize all real frames before temporal packing. Returns patches and grid_thw.
 * This API performs no frame sampling or timestamp/model-token expansion.
 * Null options uses configured geometry. Non-null options requires an initial Resize and
 * supplies exact caller-resolved geometry, overriding smart resize/aspect-ratio selection
 * without changing the processor's configuration.
 */
extError_t ORTX_API_CALL OrtxFramesPreProcess(OrtxProcessor* processor, const OrtxRawFrames* frames,
                                              const OrtxFramePreProcessOptions* options, OrtxTensorResult** result);

/**
 * @brief Loads a set of images from the specified image paths.
 *
 * This function loads a set of images from the given image paths and returns a pointer to the loaded images.
 * The number of images loaded is also returned through the `num_images_loaded` parameter.
 *
 * @param[out] images A pointer to a pointer that will be set to the loaded images.
 * @param[in] image_paths An array of image paths.
 * @param[in] num_images The number of images to load.
 * @param[out] num_images_loaded A pointer to a variable that will be set to the number of images loaded.
 *
 * @return An error code indicating the status of the operation.
 */
extError_t ORTX_API_CALL OrtxLoadImages(OrtxRawImages** images, const char** image_paths, size_t num_images,
                                        size_t* num_images_loaded);

/**
 * @brief Creates raw images from the provided data.
 *
 * This function creates raw images from the provided data. The raw images are stored in the `images` parameter.
 *
 * @param images Pointer to a pointer to the `OrtxRawImages` structure that will hold the created raw images.
 * @param data Array of pointers to the data for each image.
 * @param sizes Array of pointers to the sizes of each image.
 * @param num_images Number of images to create.
 * @return An `extError_t` value indicating the success or failure of the operation.
 */
extError_t ORTX_API_CALL OrtxCreateRawImages(OrtxRawImages** images, const void* data[], const int64_t sizes[],
                                             size_t num_images);

/**
 * @brief Pre-processes the given raw images using the specified processor.
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
                                             OrtxTensorResult** result);

/** Preprocess encoded images using one resolved geometry per image.
 * Requires a DecodeImage/ConvertRGB/Resize prefix. Decode once, verify source dimensions,
 * and resize to the exact supplied size without smart resizing or aspect-ratio selection.
 * Options count must equal image count; all geometries are validated before decoding.
 * The processor configuration remains unchanged. Failure sets *result to null.
 */
extError_t ORTX_API_CALL OrtxImagePreProcessWithOptions(OrtxProcessor* processor, OrtxRawImages* images,
                                                        const OrtxImagePreProcessOptions* options, size_t options_count,
                                                        OrtxTensorResult** result);

#ifdef __cplusplus
}
#endif
