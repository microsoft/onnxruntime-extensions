// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ortx_processor.h"
#include "image_processor.h"

using namespace ort_extensions;

extError_t OrtxCreateProcessor(OrtxProcessor** processor, const char* def) {
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
  size_t num_images;
};

extError_t ORTX_API_CALL OrtxLoadImages(OrtxRawImages** images, const char** image_paths, size_t num_images,
                                        size_t* num_images_loaded) {
  if (images == nullptr || image_paths == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto images_obj = std::make_unique<RawImagesObject>();
  auto [img, num] = LoadRawImages(image_paths, image_paths + num_images);
  images_obj->images = std::move(img);
  images_obj->num_images = num;
  if (num_images_loaded != nullptr) {
    *num_images_loaded = num;
  }
  
  *images = static_cast<OrtxRawImages*>(images_obj.release());
  return extError_t();
}

extError_t ORTX_API_CALL OrtxImagePreProcess(OrtxProcessor* processor, OrtxRawImages* images,
                                             OrtxImageProcessorResult** result) {
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

  auto result_ptr = std::make_unique<ImageProcessorResult>();
  status =
      processor_ptr->PreProcess(ort_extensions::span(images_ptr->images.get(), images_ptr->num_images), *result_ptr);
  if (status.IsOk()) {
    *result = static_cast<OrtxImageProcessorResult*>(result_ptr.release());
  } else {
    *result = nullptr;
  }

  return {};
}

extError_t ORTX_API_CALL OrtxImageGetTensorResult(OrtxImageProcessorResult* result, size_t index, OrtxTensor** tensor) {
  if (result == nullptr || tensor == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto result_ptr = static_cast<ImageProcessorResult*>(result);
  ReturnableStatus status(result_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindImageProcessorResult));
  if (!status.IsOk()) {
    return status.Code();
  }

  if (index >= result_ptr->results.size()) {
    ReturnableStatus::last_error_message_ = "Index out of range";
    return kOrtxErrorInvalidArgument;
  }

  auto tensor_ptr = std::make_unique<OrtxObjectWrapper<ortc::TensorBase>>();
  tensor_ptr->SetObject(result_ptr->results[index].get());
  *tensor = static_cast<OrtxTensor*>(tensor_ptr.release());
  return extError_t();
}

extError_t ORTX_API_CALL OrtxClearOutputs(OrtxProcessor* processor, OrtxImageProcessorResult* result) {
  if (processor == nullptr || result == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  const auto processor_ptr = static_cast<const ImageProcessor*>(processor);
  ReturnableStatus status(processor_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindProcessor));
  if (!status.IsOk()) {
    return status.Code();
  }

  auto result_ptr = static_cast<ImageProcessorResult*>(result);
  status = result_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindImageProcessorResult);
  if (!status.IsOk()) {
    return status.Code();
  }

  ImageProcessor::ClearOutputs(result_ptr);
  return extError_t();
}
