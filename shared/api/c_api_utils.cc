// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <stdio.h>
#include <cstdarg>

#include "ortx_utils.h"
#include "file_sys.h"

#include "tokenizer_impl.h"
#include "image_processor.h"
#include "speech_extractor.h"

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

extError_t ORTX_API_CALL OrtxDisposeOnly(OrtxObject* object) {
  if (object == nullptr) {
    return kOrtxErrorInvalidArgument;
  }

  auto Ortx_object = static_cast<OrtxObjectImpl*>(object);
  if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindUnknown) {
    return kOrtxErrorInvalidArgument;
  }

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

  auto tensor_ptr = std::make_unique<TensorObject>();
  tensor_ptr->SetTensor(ts);
  *tensor = static_cast<OrtxTensor*>(tensor_ptr.release());
  return extError_t();
}

extError_t ORTX_API_CALL OrtxGetTensorType(OrtxTensor* tensor, extDataType_t* type) {
  if (tensor == nullptr || type == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto tensor_impl = static_cast<TensorObject*>(tensor);
  if (tensor_impl->ortx_kind() != extObjectKind_t::kOrtxKindTensor) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  *type = tensor_impl->GetTensorType();
  return extError_t();
}

extError_t ORTX_API_CALL OrtxGetTensorSizeOfElement(OrtxTensor* tensor, size_t* size) {
  if (tensor == nullptr || size == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto tensor_impl = static_cast<TensorObject*>(tensor);
  if (tensor_impl->ortx_kind() != extObjectKind_t::kOrtxKindTensor) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto tb = tensor_impl->GetTensor();
  assert(tb != nullptr); 
  *size = tb->SizeInBytes() / tb->NumberOfElement();
  return extError_t();
}

extError_t ORTX_API_CALL OrtxGetTensorData(OrtxTensor* tensor, const void** data, const int64_t** shape,
                                           size_t* num_dims) {
  if (tensor == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto tensor_impl = static_cast<TensorObject*>(tensor);
  if (tensor_impl->ortx_kind() != extObjectKind_t::kOrtxKindTensor) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  *data = tensor_impl->GetTensor()->DataRaw();
  *shape = tensor_impl->GetTensor()->Shape().data();
  *num_dims = tensor_impl->GetTensor()->Shape().size();
  return extError_t();
}

extError_t ORTX_API_CALL OrtxGetTensorDataInt64(OrtxTensor* tensor, const int64_t** data, const int64_t** shape,
                                                size_t* num_dims) {
  const void* data_ptr{};
  auto err = OrtxGetTensorData(tensor, &data_ptr, shape, num_dims);
  *data = reinterpret_cast<const int64_t*>(data_ptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  return err;
}

extError_t ORTX_API_CALL OrtxGetTensorDataFloat(OrtxTensor* tensor, const float** data, const int64_t** shape,
                                                size_t* num_dims) {
  const void* data_ptr{};
  auto err = OrtxGetTensorData(tensor, &data_ptr, shape, num_dims);
  *data = reinterpret_cast<const float*>(data_ptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  return err;
}

extError_t ORTX_API_CALL OrtxGetTensorDataUint8(OrtxTensor* tensor, const uint8_t** data, const int64_t** shape,
                                                size_t* num_dims) {
  const void* data_ptr{};
  auto err = OrtxGetTensorData(tensor, &data_ptr, shape, num_dims);
  *data = reinterpret_cast<const uint8_t*>(data_ptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
  return err;
}
