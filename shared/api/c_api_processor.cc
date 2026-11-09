// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ortx_processor.h"
#include "image_processor.h"

#include "c_api_utils.hpp"
#ifdef OCOS_ENABLE_VIDEO_DECODER
#include "video_decoder.h"
#endif

#include <cmath>
#include <cstring>
#include <limits>

using namespace ort_extensions;

struct RawFramesObject : public OrtxObjectImpl {
  RawFramesObject() : OrtxObjectImpl(kOrtxKindRawFrames) {}
  std::vector<ImageRawData> pixels;
  std::vector<OrtxFrame> descriptors;
  std::vector<OrtxVideoSample> samples;
  OrtxVideoMetadata metadata{};
};

#ifdef OCOS_ENABLE_VIDEO_DECODER
namespace {
constexpr OrtxVideoLimits kDefaultVideoLimits{sizeof(OrtxVideoLimits),
                                              64,
                                              256ULL * 1024 * 1024,
                                              512ULL * 1024 * 1024,
                                              1024ULL * 1024 * 1024,
                                              100000,
                                              4096,
                                              4096,
                                              4096};
}
#endif

extError_t ORTX_API_CALL OrtxValidateVideoLoadOptions(const OrtxVideoLoadOptions* options,
                                                      const OrtxVideoLimits* limits) {
#ifdef OCOS_ENABLE_VIDEO_DECODER
  if (options == nullptr) {
    ReturnableStatus::last_error_message_ = "Missing video sampling options";
    return kOrtxErrorInvalidArgument;
  }
  try {
    ReturnableStatus status(ValidateVideoOptions(*options, limits == nullptr ? kDefaultVideoLimits : *limits));
    return status.Code();
  } catch (const std::bad_alloc&) {
    ReturnableStatus::last_error_message_ = "Video target validation allocation failed";
    return kOrtxErrorOutOfMemory;
  } catch (const std::exception& error) {
    ReturnableStatus::last_error_message_ = error.what();
    return kOrtxErrorInternal;
  }
#else
  (void)options;
  (void)limits;
  ReturnableStatus::last_error_message_ = "Video file decoding is unavailable in this build";
  return kOrtxErrorNotImplemented;
#endif
}

extError_t ORTX_API_CALL OrtxLoadVideo(OrtxRawFrames** frames, const char* path, const OrtxVideoLoadOptions* options,
                                       const OrtxVideoLimits* limits) {
  if (frames == nullptr) {
    ReturnableStatus::last_error_message_ = "Missing video frames output";
    return kOrtxErrorInvalidArgument;
  }
  *frames = nullptr;
#ifdef OCOS_ENABLE_VIDEO_DECODER
  if (path == nullptr || *path == '\0' || options == nullptr) {
    ReturnableStatus::last_error_message_ = "Missing video path or sampling options";
    return kOrtxErrorInvalidArgument;
  }
  try {
    auto result = std::make_unique<RawFramesObject>();
    ReturnableStatus status(DecodeVideo(path, *options, limits == nullptr ? kDefaultVideoLimits : *limits,
                                        result->pixels, result->descriptors, result->samples, result->metadata));
    if (!status.IsOk()) return status.Code();
    *frames = result.release();
    return kOrtxOK;
  } catch (const std::bad_alloc&) {
    ReturnableStatus::last_error_message_ = "Video loading allocation failed";
    return kOrtxErrorOutOfMemory;
  } catch (const std::exception& error) {
    ReturnableStatus::last_error_message_ = error.what();
    return kOrtxErrorInvalidFile;
  }
#else
  (void)path;
  (void)options;
  (void)limits;
  ReturnableStatus::last_error_message_ = "Video file decoding is unavailable in this build";
  return kOrtxErrorNotImplemented;
#endif
}

extError_t ORTX_API_CALL OrtxGetVideoMetadata(const OrtxRawFrames* frames, OrtxVideoMetadata* metadata) {
  if (frames == nullptr || metadata == nullptr || metadata->struct_size < sizeof(OrtxVideoMetadata)) {
    ReturnableStatus::last_error_message_ = "Invalid video metadata output";
    return kOrtxErrorInvalidArgument;
  }
  ReturnableStatus status(static_cast<const OrtxObjectImpl*>(frames)->IsInstanceOf(kOrtxKindRawFrames));
  if (!status.IsOk()) return status.Code();
  const auto& value = static_cast<const RawFramesObject*>(frames)->metadata;
  if (value.time_base_den == 0) {
    ReturnableStatus::last_error_message_ = "File timeline metadata is unavailable for caller-supplied frames";
    return kOrtxErrorNotImplemented;
  }
  *metadata = value;
  return kOrtxOK;
}

extError_t ORTX_API_CALL OrtxGetVideoSample(const OrtxRawFrames* frames, size_t index, OrtxVideoSample* sample) {
  if (frames == nullptr || sample == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid video sample output";
    return kOrtxErrorInvalidArgument;
  }
  ReturnableStatus status(static_cast<const OrtxObjectImpl*>(frames)->IsInstanceOf(kOrtxKindRawFrames));
  if (!status.IsOk()) return status.Code();
  const auto& samples = static_cast<const RawFramesObject*>(frames)->samples;
  if (index >= samples.size()) {
    ReturnableStatus::last_error_message_ = "Video sample index out of range";
    return kOrtxErrorOutOfRange;
  }
  *sample = samples[index];
  return kOrtxOK;
}

extError_t ORTX_API_CALL OrtxValidateRawFrames(const OrtxFrame* descriptors, size_t num_frames,
                                               const OrtxFrameLimits* limits, size_t* packed_bytes) {
  if (packed_bytes == nullptr) {
    ReturnableStatus::last_error_message_ = "Missing frame byte-count output";
    return kOrtxErrorInvalidArgument;
  }
  *packed_bytes = 0;
  const OrtxFrameLimits defaults{sizeof(OrtxFrameLimits), 64, 256ULL * 1024 * 1024, 16384};
  if (limits == nullptr) {
    limits = &defaults;
  }
  if (limits->struct_size < sizeof(OrtxFrameLimits) || limits->max_frames == 0 || limits->max_bytes == 0 ||
      limits->max_dimension <= 0 || descriptors == nullptr || num_frames == 0 || num_frames > limits->max_frames) {
    ReturnableStatus::last_error_message_ = "Invalid frame limits or frame count";
    return kOrtxErrorInvalidArgument;
  }

  size_t total_bytes = 0;
  for (size_t i = 0; i < num_frames; ++i) {
    const auto& frame = descriptors[i];
    if (frame.data == nullptr || frame.width <= 0 || frame.height <= 0 || frame.width > limits->max_dimension ||
        frame.height > limits->max_dimension || !std::isfinite(frame.timestamp_seconds) ||
        frame.timestamp_seconds < 0 || (i != 0 && frame.timestamp_seconds < descriptors[i - 1].timestamp_seconds)) {
      ReturnableStatus::last_error_message_ = "Invalid frame geometry, buffer or presentation timestamp";
      return kOrtxErrorInvalidArgument;
    }
    const auto maximum = std::numeric_limits<size_t>::max();
    if (static_cast<uint64_t>(frame.width) > maximum / 3) {
      ReturnableStatus::last_error_message_ = "Frame geometry overflows addressable storage";
      return kOrtxErrorInvalidArgument;
    }
    const size_t row_bytes = static_cast<size_t>(frame.width) * 3;
    if (static_cast<uint64_t>(frame.height) > maximum / row_bytes) {
      ReturnableStatus::last_error_message_ = "Frame geometry overflows addressable storage";
      return kOrtxErrorInvalidArgument;
    }
    const size_t height = static_cast<size_t>(frame.height);
    if (frame.row_stride < row_bytes || height - 1 > (maximum - row_bytes) / frame.row_stride ||
        frame.data_size < (height - 1) * frame.row_stride + row_bytes ||
        height > (limits->max_bytes - total_bytes) / row_bytes) {
      ReturnableStatus::last_error_message_ = "Frame extent, row stride or aggregate RGB quota exceeded";
      return kOrtxErrorInvalidArgument;
    }
    total_bytes += height * row_bytes;
  }
  *packed_bytes = total_bytes;
  return kOrtxOK;
}

extError_t ORTX_API_CALL OrtxCreateRawFrames(OrtxRawFrames** frames, const OrtxFrame* descriptors, size_t num_frames,
                                             const OrtxFrameLimits* limits) {
  if (frames == nullptr) {
    ReturnableStatus::last_error_message_ = "Missing frames output";
    return kOrtxErrorInvalidArgument;
  }
  *frames = nullptr;
  size_t total_bytes{};
  const auto status = OrtxValidateRawFrames(descriptors, num_frames, limits, &total_bytes);
  if (status != kOrtxOK) return status;

  try {
    auto result = std::make_unique<RawFramesObject>();
    result->pixels.resize(num_frames);
    result->descriptors.resize(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
      const auto& source = descriptors[i];
      const size_t row_bytes = static_cast<size_t>(source.width) * 3;
      auto& pixels = result->pixels[i];
      pixels.resize(static_cast<size_t>(source.height) * row_bytes);
      for (size_t row = 0; row < static_cast<size_t>(source.height); ++row) {
        std::memcpy(pixels.data() + row * row_bytes, static_cast<const uint8_t*>(source.data) + row * source.row_stride,
                    row_bytes);
      }
      result->descriptors[i] = {pixels.data(), pixels.size(), source.width,
                                source.height, row_bytes,     source.timestamp_seconds};
    }
    *frames = result.release();
    return kOrtxOK;
  } catch (const std::bad_alloc&) {
    ReturnableStatus::last_error_message_ = "Frame snapshot allocation failed";
    return kOrtxErrorOutOfMemory;
  } catch (const std::exception& error) {
    ReturnableStatus::last_error_message_ = error.what();
    return kOrtxErrorInternal;
  }
}

extError_t ORTX_API_CALL OrtxGetRawFrameCount(const OrtxRawFrames* frames, size_t* count) {
  if (frames == nullptr || count == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid frame count argument";
    return kOrtxErrorInvalidArgument;
  }
  ReturnableStatus status(static_cast<const OrtxObjectImpl*>(frames)->IsInstanceOf(kOrtxKindRawFrames));
  if (!status.IsOk()) {
    return status.Code();
  }
  *count = static_cast<const RawFramesObject*>(frames)->descriptors.size();
  return kOrtxOK;
}

extError_t ORTX_API_CALL OrtxGetRawFrame(const OrtxRawFrames* frames, size_t index, OrtxFrame* descriptor) {
  if (frames == nullptr || descriptor == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid frame view argument";
    return kOrtxErrorInvalidArgument;
  }
  ReturnableStatus status(static_cast<const OrtxObjectImpl*>(frames)->IsInstanceOf(kOrtxKindRawFrames));
  if (!status.IsOk()) {
    return status.Code();
  }
  const auto& descriptors = static_cast<const RawFramesObject*>(frames)->descriptors;
  if (index >= descriptors.size()) {
    ReturnableStatus::last_error_message_ = "Frame index out of range";
    return kOrtxErrorOutOfRange;
  }
  *descriptor = descriptors[index];
  return kOrtxOK;
}

extError_t ORTX_API_CALL OrtxFramesPreProcess(OrtxProcessor* processor, const OrtxRawFrames* frames,
                                              const OrtxFramePreProcessOptions* options, OrtxTensorResult** result) {
  if (result == nullptr) {
    ReturnableStatus::last_error_message_ = "Missing frame preprocessing output";
    return kOrtxErrorInvalidArgument;
  }
  *result = nullptr;
  if (processor == nullptr || frames == nullptr) {
    ReturnableStatus::last_error_message_ = "Missing frame processor or frames";
    return kOrtxErrorInvalidArgument;
  }
  if (options != nullptr &&
      (options->struct_size < sizeof(OrtxFramePreProcessOptions) || options->height <= 0 || options->width <= 0 ||
       options->height > (std::numeric_limits<int>::max)() || options->width > (std::numeric_limits<int>::max)() / 4 ||
       static_cast<uint64_t>(options->height) >
           (std::numeric_limits<size_t>::max)() / (static_cast<size_t>(options->width) * 3))) {
    ReturnableStatus::last_error_message_ = "Invalid resolved frame resize geometry";
    return kOrtxErrorInvalidArgument;
  }
  ReturnableStatus status(static_cast<const OrtxObjectImpl*>(processor)->IsInstanceOf(kOrtxKindProcessor));
  if (!status.IsOk()) {
    return status.Code();
  }
  status = static_cast<const OrtxObjectImpl*>(frames)->IsInstanceOf(kOrtxKindRawFrames);
  if (!status.IsOk()) {
    return status.Code();
  }
  try {
    const auto& descriptors = static_cast<const RawFramesObject*>(frames)->descriptors;
    auto output = std::make_unique<TensorResult>();
    status = static_cast<const ImageProcessor*>(processor)->PreProcessFrames(
        span<const OrtxFrame>(descriptors.data(), descriptors.size()), options, *output);
    if (!status.IsOk()) {
      return status.Code();
    }
    *result = output.release();
    return kOrtxOK;
  } catch (const std::bad_alloc&) {
    ReturnableStatus::last_error_message_ = "Frame preprocessing allocation failed";
    return kOrtxErrorOutOfMemory;
  } catch (const std::exception& error) {
    ReturnableStatus::last_error_message_ = error.what();
    return kOrtxErrorInternal;
  }
}

extError_t ORTX_API_CALL OrtxCreateProcessor(OrtxProcessor** processor, const char* def) {
  if (processor == nullptr || def == nullptr) {
    return kOrtxErrorInvalidArgument;
  }

  auto proc_ptr = std::make_unique<ImageProcessor>();
  ReturnableStatus status = proc_ptr->Init(def);
  if (status.IsOk()) {
    *processor = static_cast<OrtxProcessor*>(proc_ptr.release());
  } else {
    *processor = nullptr;
  }

  return status.Code();
}

struct RawImagesObject : public OrtxObjectImpl {
 public:
  RawImagesObject() : OrtxObjectImpl(kOrtxKindRawImages) {}
  std::unique_ptr<ort_extensions::ImageRawData[]> images;
  size_t num_images{};
};

extError_t ORTX_API_CALL OrtxCreateRawImages(OrtxRawImages** images, const void* data[], const int64_t sizes[],
                                             size_t num_images) {
  if (images == nullptr || data == nullptr || sizes == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto images_obj = std::make_unique<RawImagesObject>();
  images_obj->images = std::make_unique<ImageRawData[]>(num_images);
  images_obj->num_images = num_images;
  for (size_t i = 0; i < num_images; ++i) {
    images_obj->images[i].resize(static_cast<size_t>(sizes[i]));
    std::copy_n(static_cast<const uint8_t*>(data[i]), sizes[i], images_obj->images[i].data());
  }

  *images = static_cast<OrtxRawImages*>(images_obj.release());
  return extError_t();
}

extError_t ORTX_API_CALL OrtxLoadImages(OrtxRawImages** images, const char** image_paths, size_t num_images,
                                        size_t* num_images_loaded) {
  if (images == nullptr || image_paths == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto images_obj = std::make_unique<RawImagesObject>();
  auto [img, num] = LoadRawData<char const**, ImageRawData>(image_paths, image_paths + num_images);
  images_obj->images = std::move(img);
  images_obj->num_images = num;
  if (num_images_loaded != nullptr) {
    *num_images_loaded = num;
  }

  *images = static_cast<OrtxRawImages*>(images_obj.release());
  return extError_t();
}

extError_t ORTX_API_CALL OrtxImagePreProcess(OrtxProcessor* processor, OrtxRawImages* images,
                                             OrtxTensorResult** result) {
  if (processor == nullptr || images == nullptr || result == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto processor_ptr = static_cast<ImageProcessor*>(processor);
  ReturnableStatus status(processor_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindProcessor));
  if (!status.IsOk()) {
    return status.Code();
  }

  auto images_ptr = static_cast<RawImagesObject*>(images);
  status = images_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindRawImages);
  if (!status.IsOk()) {
    return status.Code();
  }

  auto result_ptr = std::make_unique<TensorResult>();
  status =
      processor_ptr->PreProcess(ort_extensions::span(images_ptr->images.get(), images_ptr->num_images), *result_ptr);
  if (status.IsOk()) {
    *result = static_cast<OrtxTensorResult*>(result_ptr.release());
  } else {
    *result = nullptr;
  }

  return status.Code();
}

extError_t ORTX_API_CALL OrtxImagePreProcessWithOptions(OrtxProcessor* processor, OrtxRawImages* images,
                                                        const OrtxImagePreProcessOptions* options, size_t options_count,
                                                        OrtxTensorResult** result) {
  if (result == nullptr) return kOrtxErrorInvalidArgument;
  *result = nullptr;
  if (processor == nullptr || images == nullptr || (options_count != 0 && options == nullptr)) {
    ReturnableStatus::last_error_message_ = "Missing image processor, images or resolved geometry";
    return kOrtxErrorInvalidArgument;
  }
  ReturnableStatus status(static_cast<const OrtxObjectImpl*>(processor)->IsInstanceOf(kOrtxKindProcessor));
  if (!status.IsOk()) return status.Code();
  status = static_cast<const OrtxObjectImpl*>(images)->IsInstanceOf(kOrtxKindRawImages);
  if (!status.IsOk()) return status.Code();
  const auto* input = static_cast<const RawImagesObject*>(images);
  if (options_count != input->num_images) {
    ReturnableStatus::last_error_message_ = "Resolved geometry count differs from image count";
    return kOrtxErrorInvalidArgument;
  }
  for (size_t image = 0; image < options_count; ++image) {
    const auto& size = options[image];
    if (size.struct_size != sizeof(OrtxImagePreProcessOptions) || size.source_height <= 0 || size.source_width <= 0 ||
        size.height <= 0 || size.width <= 0 || size.source_height > (std::numeric_limits<int>::max)() ||
        size.source_width > (std::numeric_limits<int>::max)() / 4 || size.height > (std::numeric_limits<int>::max)() ||
        static_cast<uint64_t>(size.source_height) >
            (std::numeric_limits<size_t>::max)() / (static_cast<size_t>(size.source_width) * 3) ||
        size.height > (std::numeric_limits<int>::max)() || size.width > (std::numeric_limits<int>::max)() / 4 ||
        static_cast<uint64_t>(size.height) >
            (std::numeric_limits<size_t>::max)() / (static_cast<size_t>(size.width) * 3)) {
      ReturnableStatus::last_error_message_ = "Invalid resolved image resize geometry";
      return kOrtxErrorInvalidArgument;
    }
  }
  try {
    auto output = std::make_unique<TensorResult>();
    status = static_cast<const ImageProcessor*>(processor)->PreProcess(
        span<ImageRawData>(input->images.get(), input->num_images), *output,
        span<const OrtxImagePreProcessOptions>(options, options_count));
    if (!status.IsOk()) return status.Code();
    *result = output.release();
    return kOrtxOK;
  } catch (const std::bad_alloc&) {
    ReturnableStatus::last_error_message_ = "Image preprocessing allocation failed";
    return kOrtxErrorOutOfMemory;
  } catch (const std::exception& error) {
    ReturnableStatus::last_error_message_ = error.what();
    return kOrtxErrorInternal;
  }
}
