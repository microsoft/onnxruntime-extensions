// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ortx_utils.h"

namespace ort_extensions {

template <typename T>
class OrtxDeleter {
 public:
  void operator()(T* p) const {
    if (p) {
      OrtxDisposeOnly(p);
    }
  }
};


/**
 * @brief A smart pointer class that manages the lifetime of an OrtxObject.
 * 
 * This class is derived from std::unique_ptr and provides additional functionality
 * specific to OrtxObject. It automatically calls the OrtxDeleter to release the
 * owned object when it goes out of scope.
 * 
 * @tparam T The type of the object being managed.
 */
template <typename T>
class OrtxObjectPtr : public std::unique_ptr<T, OrtxDeleter<T>> {
 public:
  /**
   * @brief Default constructor.
   * 
   * Constructs an OrtxObjectPtr with a null pointer.
   */
  explicit OrtxObjectPtr(T* ptr=nullptr) : std::unique_ptr<T, OrtxDeleter<T>>(ptr) {}

  /**
   * @brief Constructor that creates an OrtxObjectPtr from a function call.
   * 
   * This constructor calls the specified function with the given arguments to
   * create an OrtxObject. If the function call succeeds, the created object is
   * owned by the OrtxObjectPtr.
   * 
   * @tparam TFn The type of the function pointer or function object.
   * @tparam Args The types of the arguments to be passed to the function.
   * @param fn The function pointer or function object used to create the OrtxObject.
   * @param args The arguments to be passed to the function.
   */
  template <typename TFn, typename... Args>
  OrtxObjectPtr(TFn fn, Args&&... args) {
    OrtxObject* proc = nullptr;
    err_ = fn(&proc, std::forward<Args>(args)...);
    if (err_ == kOrtxOK) {
      this->reset(static_cast<T*>(proc));
    }
  }

  template <typename TFn, typename... Args>
  static OrtxObjectPtr<T> FromCapi(TFn fn, Args&&... args) {
    OrtxObject* proc = nullptr;
    extError_t err = fn(&proc, std::forward<Args>(args)...);
    if (err == kOrtxOK) {
      return OrtxObjectPtr(static_cast<T*>(proc));
    }
  }

  /**
   * @brief Get the error code associated with the creation of the OrtxObject.
   * 
   * @return The error code.
   */
  extError_t Code() const { return err_; }

  struct PointerAssigner {
    OrtxObject* obj_{};
    OrtxObjectPtr<T>& ptr_;
    PointerAssigner(OrtxObjectPtr<T>& ptr) : ptr_(ptr){};

    ~PointerAssigner() { ptr_.reset(static_cast<T*>(obj_)); };

    operator T**() { return reinterpret_cast<T**>(&obj_); };
  };

/**
 * @brief A wrapper function for OrtxObjectPtr that can be used as a function parameter of T**.
 * 
 * This function creates a PointerAssigner object for the given OrtxObjectPtr. The PointerAssigner
 * object can be used to assign a pointer value to the OrtxObjectPtr.
 * 
 */
  PointerAssigner ToBeAssigned() { return PointerAssigner{*this}; }

 private:
  extError_t err_ = kOrtxOK; /**< The error code associated with the creation of the OrtxObject. */
};

}  // namespace ort_extensions
