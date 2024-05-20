// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>

#include "ortx_utils.h"
#include "ext_status.h"
#include "op_def_struct.h"

namespace ort_extensions {
class OrtxObjectImpl : public OrtxObject {
 public:
  explicit OrtxObjectImpl(extObjectKind_t kind = extObjectKind_t::kOrtxKindUnknown) : OrtxObject() {
    ext_kind_ = static_cast<int>(kind);
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

  void SetTokenIds(std::vector<std::vector<extTokenId_t>>&& token_ids) {
    token_ids_ = token_ids;
  }

  [[nodiscard]] const std::vector<std::vector<extTokenId_t>>& token_ids() const {
    return token_ids_;
  }

 private:
  std::vector<std::vector<extTokenId_t>> token_ids_;
};

class StringArray : public OrtxObjectImpl {
 public:
  StringArray() : OrtxObjectImpl(extObjectKind_t::kOrtxKindStringArray) {}
  ~StringArray() override = default;

  void SetStrings(std::vector<std::string>&& strings) {
    strings_ = strings;
  }

  [[nodiscard]] const std::vector<std::string>& strings() const {
    return strings_;
  }

 private:
  std::vector<std::string> strings_;
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

template <typename T>
class OrtxObjectFactory {
  public:
  static std::unique_ptr<T> Create() {
    return std::make_unique<T>();
  }

  static OrtxObject* CreateForward();
  static void DisposeForward(OrtxObject* object);

  static void Dispose(OrtxObject* object) {
    auto obj_ptr = static_cast<T*>(object);
    std::unique_ptr<T> ptr(obj_ptr);
    ptr.reset();
  }

};

class DetokenizerCache;  // forward definition in tokenizer_impl.cc
class ProcessorResult;  // forward definition in image_processor.h

template <typename T>
class OrtxDeleter {
 public:
  void operator()(T* p) const {
    if (p) {
      OrtxDisposeOnly(p);
    }
  }
};

template <typename T>
class OrtxObjectPtr : public std::unique_ptr<T, OrtxDeleter<T>> {
 public:
  template <typename TFn>
  OrtxObjectPtr(TFn fn, const char* def) {
    OrtxObject* proc = nullptr;
    err_ = fn(&proc, def);
    if (err_ == kOrtxOK) {
      this->reset(static_cast<T*>(proc));
    }
  }

  int err_ = kOrtxOK;
};

class CppAllocator : public ortc::IAllocator {
 public:
  void* Alloc(size_t size) override {
    return std::make_unique<char[]>(size).release();
  }

  void Free(void* p) override {
    std::unique_ptr<char[]> ptr(static_cast<char*>(p));
    ptr.reset();
  }

  static CppAllocator& Instance() {
    static CppAllocator allocator;
    return allocator;
  }
};

}  // namespace ort_extensions
