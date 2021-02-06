// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <vector>
#include "utils.h"

typedef enum SparseInTensorType {
  RAGGED = 1  // Follows Tensorflow ragged tensor definition.
} SparseInTensorType;

/*
* This structure serializes a sparse tensor into a single buffer.
* T is a simple type (no string).
*/
template <typename T>
struct SparseInTensor {
  inline int64_t size() const { return *size_; }                      // total number of bytes
  inline size_t ndims() const { return *ndims_; }                     // number of dimensions
  inline size_t nindices() const { return *nindices_; }               // number of indices
  inline size_t nvalues() const { return indices_[nindices() - 1]; }  // number of values

  /* Returns the shape as a copy. */
  inline std::vector<int64_t> shape() const { return std::vector<int64_t>(shape_, shape_ + ndims()); }
  /* Returns the indices as a copy. */
  inline std::vector<int64_t> indices() const { return std::vector<int64_t>(indices_, indices_ + nindices()); }
  /* Returns the values as a copy. */
  inline std::vector<T> values() const { return std::vector<T>(values_, values_ + nvalues()); }
  /* Returns a const reference on the serialized buffer. */
  inline const uint8_t* buffer() const { return buffer_; }

  /** Initialize the structure from a buffer. */
  static void create_from_buffer(const uint8_t* buffer, SparseInTensor<T>& out, bool copy = false) {
    SparseInTensorType type = (SparseInTensorType)(*(const uint32_t*)buffer);
    switch (type) {
      case SparseInTensorType::RAGGED:
        create_from_buffer_as_ragged(buffer, out, copy);
        return;
      default:
        throw std::runtime_error(MakeString("Not implemented for type ", type, "."));
    }
  }

  /** Initialize the structure from a tensorflow ragged tensor. */
  static void create_ragged(const std::vector<T>& values, const std::vector<int64_t>& indices,
                            SparseInTensor<T>& out) {
    if (indices.size() <= 1) {
      out.clear();
      return;
    }
    out.init(SparseInTensorType::RAGGED, 2, indices.size(), values.size(), nullptr);
    out.shape_[0] = indices.size() - 1;
    out.shape_[1] = 0;
    memcpy(out.indices_, indices.data(), indices.size() * sizeof(int64_t));
    memcpy(out.values_, values.data(), values.size() * sizeof(T));
  }

  /** Initialize the structure from a dense tensor. */
  static void create_ragged_from_dense(const std::vector<int64_t>& shape, const T* values,
                                       SparseInTensor<T>& out) {
    if (shape.size() == 0) {
      out.clear();
      return;
    }
    if (shape.size() != 2) {
      throw std::runtime_error(MakeString(
          "Can only create a ragged tensor from a 2D tensor, this one has ",
          shape.size(), " dimensions."));
    }
    out.init(SparseInTensorType::RAGGED, 2, shape[0] + 1, shape[0] * shape[1], nullptr);
    out.shape_[0] = shape[0];
    out.shape_[1] = shape[1];

    int64_t* ind = out.indices_;
    for (int64_t i = 0; i < shape[0]; ++i, ++ind) {
      *ind = i * shape[1];
    }
    *ind = shape[0] * shape[1];

    memcpy(out.values_, values, out.nvalues() * sizeof(T));
  }

 private:
  void clear() {
    type_ = nullptr;
    ndims_ = nullptr;
    size_ = nullptr;
    nindices_ = nullptr;
    shape_ = nullptr;
    values_ = nullptr;
    indices_ = nullptr;
    buffer_ = nullptr;
    content_.clear();
  }

  /* Allocates a new buffer, initializes all pointers pointing into that buffer. */
  void init(SparseInTensorType type, size_t ndims, int64_t nindices, int64_t nvalues, const uint8_t* buffer) {
    int64_t total_size = sizeof(uint32_t) + sizeof(size_t) + sizeof(int64_t) * ndims +
                         sizeof(int64_t) + sizeof(int64_t) + sizeof(int64_t) * nindices +
                         sizeof(T) * nvalues;
    uint8_t* ptr;
    if (buffer == nullptr) {
      content_.resize(total_size);
      ptr = content_.data();
      buffer_ = ptr;

      type_ = (uint32_t*)ptr;
      *type_ = (uint32_t)type;
      ptr += sizeof(uint32_t);

      ndims_ = (size_t*)ptr;
      *ndims_ = ndims;
      ptr += sizeof(size_t);

      size_ = (int64_t*)ptr;
      *size_ = total_size;
      ptr += sizeof(int64_t);

      nindices_ = (int64_t*)ptr;
      *nindices_ = nindices;
      ptr += sizeof(int64_t);

      shape_ = (int64_t*)ptr;
      ptr += sizeof(int64_t) * ndims;

      indices_ = (int64_t*)ptr;
      ptr += sizeof(int64_t) * nindices;

      values_ = (T*)ptr;
    } else {
      buffer_ = (uint8_t*)buffer;
      ptr = buffer_;

      type_ = (uint32_t*)ptr;
      ptr += sizeof(uint32_t);

      ndims_ = (size_t*)ptr;
      ptr += sizeof(size_t);

      size_ = (int64_t*)ptr;
      ptr += sizeof(int64_t);

      nindices_ = (int64_t*)ptr;
      ptr += sizeof(int64_t);

      shape_ = (int64_t*)ptr;
      ptr += sizeof(int64_t) * ndims;

      indices_ = (int64_t*)ptr;
      ptr += sizeof(int64_t) * nindices;

      values_ = (T*)ptr;
    }
  }

  static void create_from_buffer_as_ragged(const uint8_t* buffer, SparseInTensor<T>& out, bool copy) {
    out.clear();
    const uint8_t* head = buffer;
    SparseInTensorType type = (SparseInTensorType)(*(const uint32_t*)buffer);
    buffer += sizeof(uint32_t);
    size_t ndims = *((size_t*)buffer);
    buffer += sizeof(size_t);
    int64_t size = *((int64_t*)buffer);
    buffer += sizeof(int64_t);
    int64_t nindices = *((int64_t*)buffer);
    buffer += sizeof(int64_t);
    std::vector<int64_t> shape((int64_t*)buffer, ((int64_t*)buffer) + ndims);
    buffer += sizeof(int64_t) * ndims;
    int64_t* indices = (int64_t*)buffer;
    int64_t nvalues = indices[nindices - 1];
    if (copy) {
      out.init(type, ndims, nindices, nvalues, nullptr);
      memcpy(out.buffer_, head, size);
    } else {
      out.init(type, ndims, nindices, nvalues, head);
    }
    if (out.size() != size) {
      throw std::runtime_error(MakeString(
          "Size mismatch: ", size, " expected but size is ", out.size(), "."));
    }
  }

  uint32_t* type_;
  size_t* ndims_;
  int64_t* size_;
  int64_t* shape_;
  int64_t* nindices_;
  int64_t* indices_;
  T* values_;
  uint8_t* buffer_;
  std::vector<uint8_t> content_;
};