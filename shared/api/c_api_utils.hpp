// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <fstream>

#include "ortx_utils.h"
#include "file_sys.h"
#include "ext_status.h"
#include "op_def_struct.h"

namespace ort_extensions {
class OrtxObjectImpl : public OrtxObject {
 public:
  explicit OrtxObjectImpl(extObjectKind_t kind = extObjectKind_t::kOrtxKindUnknown) : OrtxObject() {
    ext_kind_ = kind;
  };
  virtual ~OrtxObjectImpl() = default;

  [[nodiscard]] OrtxStatus IsInstanceOf(extObjectKind_t kind) const;
  [[nodiscard]] extObjectKind_t ortx_kind() const {
    if (ext_kind_ < static_cast<int>(extObjectKind_t::kOrtxKindBegin) ||
        ext_kind_ >= static_cast<int>(extObjectKind_t::kOrtxKindEnd)) {
      return extObjectKind_t::kOrtxKindUnknown;
    }
    return static_cast<extObjectKind_t>(ext_kind_);
  }
};

// A wrapper class to store a object pointer which is readonly. i.e. unowned.
template <typename T, extObjectKind_t kind>
class OrtxObjectWrapper : public OrtxObjectImpl {
 public:
  OrtxObjectWrapper() : OrtxObjectImpl(kind) {}
  ~OrtxObjectWrapper() override = default;

  void SetObject(const T* t) { stored_object_ = t; }

  [[nodiscard]] const T* GetObject() const { return stored_object_; }

 private:
  const T* stored_object_{};
};

template <typename T>
class span {
 public:
  using value_type = std::remove_cv_t<T>;

  span(T* d, size_t s) : data_(d), size_(s) {}
  span(std::vector<value_type>& v) {
    data_ = v.data();
    size_ = v.size();
  }

  const T& operator[](size_t i) const { return data_[i]; }
  T& operator[](size_t i) { return data_[i]; }

  T* data() const { return data_; }
  [[nodiscard]] size_t size() const { return size_; }
  T* begin() const { return data_; }
  T* end() const { return data_ + size_; }

 private:
  T* data_;
  size_t size_;
};

class TokenId2DArray : public OrtxObjectImpl {
 public:
  TokenId2DArray() : OrtxObjectImpl(extObjectKind_t::kOrtxKindTokenId2DArray) {}
  ~TokenId2DArray() override = default;

  void SetTokenIds(std::vector<std::vector<extTokenId_t>>&& token_ids) { token_ids_ = token_ids; }

  [[nodiscard]] const std::vector<std::vector<extTokenId_t>>& token_ids() const { return token_ids_; }

 private:
  std::vector<std::vector<extTokenId_t>> token_ids_;
};

class StringArray : public OrtxObjectImpl {
 public:
  StringArray() : OrtxObjectImpl(extObjectKind_t::kOrtxKindStringArray) {}
  ~StringArray() override = default;

  void SetStrings(std::vector<std::string>&& strings) { strings_ = strings; }

  [[nodiscard]] const std::vector<std::string>& strings() const { return strings_; }

 private:
  std::vector<std::string> strings_;
};

class TensorObject : public OrtxObjectImpl {
 public:
  TensorObject() : OrtxObjectImpl(extObjectKind_t::kOrtxKindTensor) {}
  ~TensorObject() override = default;

  void SetTensor(ortc::TensorBase* tensor) { tensor_ = tensor; }

  static extDataType_t GetDataType(ONNXTensorElementDataType dt) {
    if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return extDataType_t::kOrtxFloat;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      return extDataType_t::kOrtxUint8;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
      return extDataType_t::kOrtxInt8;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
      return extDataType_t::kOrtxUint16;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
      return extDataType_t::kOrtxInt16;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      return extDataType_t::kOrtxInt32;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      return extDataType_t::kOrtxInt64;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      return extDataType_t::kOrtxString;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      return extDataType_t::kOrtxBool;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      return extDataType_t::kOrtxFloat16;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
      return extDataType_t::kOrtxDouble;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) {
      return extDataType_t::kOrtxUint32;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64) {
      return extDataType_t::kOrtxUint64;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64) {
      return extDataType_t::kOrtxComplex64;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128) {
      return extDataType_t::kOrtxComplex128;
    } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      return extDataType_t::kOrtxBFloat16;
    } else {
      return extDataType_t::kOrtxUnknownType;
    }
  }
  
  [[nodiscard]] extDataType_t GetTensorType() const {
    if (tensor_ == nullptr) {
      return extDataType_t::kOrtxUnknownType;
    }
    return GetDataType(tensor_->Type());
  }

  [[nodiscard]] ortc::TensorBase* GetTensor() const { return tensor_; }

 private:
  ortc::TensorBase* tensor_{};
};

class TensorResult : public OrtxObjectImpl {
 public:
  TensorResult() : OrtxObjectImpl(extObjectKind_t::kOrtxKindTensorResult) {}
  ~TensorResult() override = default;

  void SetTensors(std::vector<std::unique_ptr<ortc::TensorBase>>&& tensors) { tensors_ = std::move(tensors); }
  [[nodiscard]] size_t NumTensors() const { return tensors_.size(); }
  [[nodiscard]] const std::vector<std::unique_ptr<ortc::TensorBase>>& tensors() const { return tensors_; }
  [[nodiscard]] std::vector<ortc::TensorBase*> GetTensors() const {
    std::vector<ortc::TensorBase*> ts;
    ts.reserve(tensors_.size());
    for (auto& t : tensors_) {
      ts.push_back(t.get());
    }
    return ts;
  }

  ortc::TensorBase* GetAt(size_t i) const {
    if (i < tensors_.size()) {
      return tensors_[i].get();
    }
    return nullptr;
  }

 private:
  std::vector<std::unique_ptr<ortc::TensorBase>> tensors_;
};

struct ReturnableStatus {
 public:
  thread_local static std::string last_error_message_;

  ReturnableStatus() = default;
  ReturnableStatus(OrtxStatus&& status) : status_(status) {}
  ~ReturnableStatus() {
    if (!status_.IsOk()) {
      last_error_message_ = status_.Message();
    }
  }

  ReturnableStatus& operator=(OrtxStatus&& status) {
    status_ = status;
    return *this;
  }
  bool IsOk() const { return status_.IsOk(); }
  extError_t Code() const { return status_.Code(); }

 private:
  OrtxStatus status_;
};

class OrtxObjectFactory {
 public:
  template <typename T>
  static OrtxObject* Create() {
    return std::make_unique<T>().release();
  }

  template <typename T>
  static void Dispose(OrtxObject* object) {
    auto obj_ptr = static_cast<T*>(object);
    std::unique_ptr<T> ptr(obj_ptr);
    ptr.reset();
  }

  // Forward declaration for creating an object which isn't visible to c_api_utils.cc
  // and the definition is in the corresponding .cc file.
  template <typename T>
  static OrtxObject* CreateForward();
};

class CppAllocator : public ortc::IAllocator {
 public:
  void* Alloc(size_t size) override { return std::make_unique<char[]>(size).release(); }

  void Free(void* p) override {
    std::unique_ptr<char[]> ptr(static_cast<char*>(p));
    ptr.reset();
  }

  static CppAllocator& Instance() {
    static CppAllocator allocator;
    return allocator;
  }
};

template <typename It, typename T>
std::tuple<std::unique_ptr<T[]>, size_t> LoadRawData(It begin, It end) {
  auto raw_data = std::make_unique<T[]>(end - begin);
  size_t n = 0;
  for (auto it = begin; it != end; ++it) {
    std::ifstream ifs = path(*it).open(std::ios::binary | std::ios::in);
    if (!ifs.is_open()) {
      break;
    }

    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    T& datum = raw_data[n++];
    datum.resize(size);
    ifs.read(reinterpret_cast<char*>(datum.data()), size);
  }

  return std::make_tuple(std::move(raw_data), n);
}
}  // namespace ort_extensions
