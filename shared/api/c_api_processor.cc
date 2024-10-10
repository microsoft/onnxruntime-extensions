// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ortx_processor.h"
#include "image_processor.h"

#include "c_api_utils.hpp"

using namespace ort_extensions;

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

extError_t ORTX_API_CALL
OrtxCreateRawImages(OrtxRawImages** images, const void* data[], const int64_t sizes[], size_t num_images) {
  if (images == nullptr || data == nullptr || sizes == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto images_obj = std::make_unique<RawImagesObject>();
  images_obj->images = std::make_unique<ImageRawData[]>(num_images);
  for (size_t i = 0; i < num_images; ++i) {
    images_obj->images[i].resize(static_cast<size_t>(sizes[i]));
    std::copy_n(static_cast<const uint8_t*>(data[i]), sizes[i], images_obj->images[i].data());
  }

  return {};
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
