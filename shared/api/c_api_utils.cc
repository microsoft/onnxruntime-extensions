// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cstdarg>

#include "image_processor.h"
#include "tokenizer_impl.h"

using namespace ort_extensions;

thread_local std::string ReturnableStatus::last_error_message_;

OrtxStatus OrtxObjectImpl::IsInstanceOf(extObjectKind_t kind) const {
  if (ext_kind_ != static_cast<int>(kind)) {
    return {extError_t::kOrtxErrorInvalidArgument,
            "Object is not an instance of the requested type"};
  }
  return {};
}

int ORTX_API_CALL OrtxGetAPIVersion() {
  return API_VERSION;
}

const char* OrtxGetLastErrorMessage() {
  return ReturnableStatus::last_error_message_.c_str();
}

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
    *object = OrtxObjectFactory<DetokenizerCache>::CreateForward();
  } else  if (kind == extObjectKind_t::kOrtxKindTokenizer) {
    return OrtxCreateTokenizer(static_cast<OrtxTokenizer**>(object), va_arg(args, const char*));
  }

  va_end(args);
  return extError_t();
}

extError_t ORTX_API_CALL OrtxCreateTokenizer(OrtxTokenizer** tokenizer,
                                             const char* tokenizer_path) {
  // test if the tokenizer_path is a valid directory
  if (tokenizer_path == nullptr) {
    ReturnableStatus::last_error_message_ = "The tokenizer data directory is null";
    return kOrtxErrorInvalidArgument;
  }

  if (!std::filesystem::is_directory(tokenizer_path)) {
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

extError_t ORTX_API_CALL OrtxDispose(OrtxObject** object) {
  if (object == nullptr || *object == nullptr) {
    return kOrtxErrorInvalidArgument;
  }

  auto Ortx_object = static_cast<OrtxObjectImpl*>(*object);
  if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindUnknown) {
    return kOrtxErrorInvalidArgument;
  }

  if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindStringArray) {
    OrtxObjectFactory<ort_extensions::StringArray>::Dispose(*object);
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindTokenId2DArray) {
    OrtxObjectFactory<ort_extensions::TokenId2DArray>::Dispose(*object);
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindDetokenizerCache) {
    OrtxObjectFactory<ort_extensions::DetokenizerCache>::DisposeForward(*object);
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindTokenizer) {
    OrtxObjectFactory<ort_extensions::TokenizerImpl>::Dispose(*object);
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindProcessor) {
    OrtxObjectFactory<ort_extensions::ImageProcessor>::Dispose(*object);
  }

  *object = nullptr;
  return extError_t();
}

extError_t ORTX_API_CALL OrtxDisposeOnly(OrtxObject* object) {
  return OrtxDispose(&object);
}
