// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <stdio.h>
#include <cstdarg>

#include "file_sys.h"
#include "image_processor.h"
#include "tokenizer_impl.h"
#include "ortx_utils.h"

using namespace ort_extensions;

class DetokenizerCache;  // forward definition in tokenizer_impl.cc

thread_local std::string ReturnableStatus::last_error_message_;

OrtxStatus OrtxObjectImpl::IsInstanceOf(extObjectKind_t kind) const {
  if (ext_kind_ != static_cast<int>(kind)) {
    return {extError_t::kOrtxErrorInvalidArgument, "Object is not an instance of the requested type"};
  }
  return {};
}

int ORTX_API_CALL OrtxGetAPIVersion() { return API_VERSION; }

const char* OrtxGetLastErrorMessage() { return ReturnableStatus::last_error_message_.c_str(); }

extError_t ORTX_API_CALL OrtxCreate(extObjectKind_t kind, OrtxObject** object, ...) {
  if (object == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  if (kind == extObjectKind_t::kOrtxKindUnknown) {
    return kOrtxErrorInvalidArgument;
  }

  va_list args;
  va_start(args, object);

  if (kind == extObjectKind_t::kOrtxKindDetokenizerCache) {
    *object = OrtxObjectFactory::CreateForward<DetokenizerCache>();
  } else if (kind == extObjectKind_t::kOrtxKindTokenizer) {
    return OrtxCreateTokenizer(static_cast<OrtxTokenizer**>(object), va_arg(args, const char*));
  }

  va_end(args);
  return extError_t();
}

extError_t ORTX_API_CALL OrtxCreateTokenizer(OrtxTokenizer** tokenizer, const char* tokenizer_path) {
  // test if the tokenizer_path is a valid directory
  if (tokenizer_path == nullptr) {
    ReturnableStatus::last_error_message_ = "The tokenizer data directory is null";
    return kOrtxErrorInvalidArgument;
  }

  if (!path(tokenizer_path).is_directory()) {
    ReturnableStatus::last_error_message_ = std::string("Cannot find the directory of ") + tokenizer_path;
    return kOrtxErrorInvalidArgument;
  }

  ReturnableStatus status;
  // auto ptr = ort_extensions::CreateTokenizer(tokenizer_path, "", &status);
  auto ptr = std::make_unique<ort_extensions::TokenizerImpl>();
  status = ptr->Load(tokenizer_path);
  if (status.IsOk()) {
    *tokenizer = static_cast<OrtxTokenizer*>(ptr.release());
    return extError_t();
  }

  return status.Code();
}

extError_t ORTX_API_CALL OrtxDisposeOnly(OrtxObject* object) {
  if (object == nullptr) {
    return kOrtxErrorInvalidArgument;
  }

  auto Ortx_object = static_cast<OrtxObjectImpl*>(object);
  if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindUnknown) {
    return kOrtxErrorInvalidArgument;
  }

  /* if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindStringArray) {
    OrtxObjectFactory::Dispose<StringArray>(object);
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindTokenId2DArray) {
    OrtxObjectFactory<TokenId2DArray>::Dispose(object);
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindDetokenizerCache) {
    OrtxObjectFactory<DetokenizerCache>::DisposeForward(object);
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindTokenizer) {
    OrtxObjectFactory<TokenizerImpl>::Dispose(object);
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindProcessorResult) {
    OrtxObjectFactory<ProcessorResult>::Dispose(object);
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindImageProcessorResult) {
    OrtxObjectFactory<ImageProcessorResult>::Dispose(object);
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindProcessor) {
    OrtxObjectFactory<ImageProcessor>::Dispose(object);
  } */
  if (Ortx_object->ortx_kind() >= kOrtxKindBegin && Ortx_object->ortx_kind() < kOrtxKindEnd) {
    OrtxObjectFactory::Dispose<OrtxObjectImpl>(object);
  } else {
    return kOrtxErrorInvalidArgument;
  }

  return extError_t();
}

extError_t ORTX_API_CALL OrtxDispose(OrtxObject** object) {
  if (object == nullptr) {
    return kOrtxErrorInvalidArgument;
  }

  extError_t err = OrtxDisposeOnly(*object);
  if (err != extError_t()) {
    return err;
  }

  *object = nullptr;
  return err;
}

extError_t ORTX_API_CALL OrtxTensorResultGetAt(OrtxTensorResult* result, size_t index, OrtxTensor** tensor) {
  if (result == nullptr || tensor == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto result_ptr = static_cast<TensorResult*>(result);
  ReturnableStatus status(result_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindTensorResult));
  if (!status.IsOk()) {
    return status.Code();
  }

  ortc::TensorBase* ts = result_ptr->GetAt(index);
  if (ts == nullptr) {
    ReturnableStatus::last_error_message_ = "Cannot get the tensor at the specified index from the result";
    return kOrtxErrorInvalidArgument;
  }

  auto tensor_ptr = std::make_unique<OrtxObjectWrapper<ortc::TensorBase, kOrtxKindTensor>>();
  tensor_ptr->SetObject(ts);
  *tensor = static_cast<OrtxTensor*>(tensor_ptr.release());
  return extError_t();
}

extError_t ORTX_API_CALL OrtxGetTensorData(OrtxTensor* tensor, const void** data, const int64_t** shape,
                                           size_t* num_dims) {
  if (tensor == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto tensor_impl = static_cast<OrtxObjectWrapper<ortc::TensorBase, kOrtxKindTensor>*>(tensor);
  if (tensor_impl->ortx_kind() != extObjectKind_t::kOrtxKindTensor) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  *data = tensor_impl->GetObject()->DataRaw();
  *shape = tensor_impl->GetObject()->Shape().data();
  *num_dims = tensor_impl->GetObject()->Shape().size();
  return extError_t();
}

extError_t ORTX_API_CALL OrtxGetTensorDataInt64(OrtxTensor* tensor, const int64_t** data, const int64_t** shape,
                                                size_t* num_dims) {
  const void* data_ptr;
  auto err = OrtxGetTensorData(tensor, &data_ptr, shape, num_dims);
  *data = reinterpret_cast<const int64_t*>(data_ptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  return err;
}

extError_t ORTX_API_CALL OrtxGetTensorDataFloat(OrtxTensor* tensor, const float** data, const int64_t** shape,
                                                size_t* num_dims) {
  const void* data_ptr;
  auto err = OrtxGetTensorData(tensor, &data_ptr, shape, num_dims);
  *data = reinterpret_cast<const float*>(data_ptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  return err;
}
