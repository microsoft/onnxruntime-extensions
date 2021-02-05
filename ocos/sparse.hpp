// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <vector>
#include "utils.h"

typedef enum _SparseInTensorType {
  RAGGED = 1
} SparseInTensorType;

template <typename T>
struct SparseInTensor {
  using value_type = typename T;

  inline int64_t size() const { return content_.size(); }
  inline size_t ndims() const { return *ndims_; }
  inline size_t nindices() const { return *nindices_; }
  inline size_t nvalues() const { return indices_[nindices() - 1]; }
  inline std::vector<int64_t> shape() const { return std::vector<int64_t>(shape_, shape_ + ndims()); }
  inline std::vector<int64_t> indices() const { return std::vector<int64_t>(indices_, indices_ + nindices()); }
  inline std::vector<T> values() const { return std::vector<T>(values_, values_ + nvalues()); }
  inline const std::vector<uint8_t>& get_content() const { return content_; }

  static void create_ragged(const std::vector<T>& values, const std::vector<int64_t>& indices,
                            SparseInTensor<T>& out) {
    if (indices.size() <= 1) {
      out.clear();
      return;
    }
    out.init(SparseInTensorType::RAGGED, 2, indices.size(), values.size());
    out.shape_[0] = indices.size() - 1;
    out.shape_[1] = 0;
    memcpy(out.indices_, indices.data(), indices.size() * sizeof(int64_t));
    memcpy(out.values_, values.data(), values.size() * sizeof(T));
  }

  static void create_ragged_from_dense(const std::vector<int64_t>& shape, const std::vector<T>& values,
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
    out.init(SparseInTensorType::RAGGED, 2, shape[0] + 1, values.size());
    out.shape_[0] = shape[0];
    out.shape_[1] = shape[1];
    memcpy(out.values_, values.data(), values.size() * sizeof(T));

    int64_t* ind = out.indices_;
    for (int64_t i = 0; i < shape[0]; ++i, ++ind) {
      *ind = i * shape[1];
    }
    *ind = shape[0] * shape[1];
  }

 private:
  void clear() {
    type_ = nullptr;
    ndims_ = nullptr;
    nindices_ = nullptr;
    shape_ = nullptr;
    values_ = nullptr;
    indices_ = nullptr;
    content_.clear();
  }

  void init(SparseInTensorType type, size_t ndims, int64_t nindices, int64_t nvalues) {
    int64_t total_size = sizeof(uint32_t) + sizeof(size_t) + sizeof(int64_t) * ndims +
                         sizeof(int64_t) + sizeof(int64_t) * nindices + sizeof(T) * nvalues;
    content_.resize(total_size);
    uint8_t* ptr = content_.data();

    type_ = (uint32_t*)ptr;
    *type_ = (uint32_t)type;
    ptr += sizeof(uint32_t);

    ndims_ = (size_t*)ptr;
    *ndims_ = ndims;
    ptr += sizeof(size_t);

    nindices_ = (int64_t*)ptr;
    *nindices_ = nindices;
    ptr += sizeof(int64_t);

    shape_ = (int64_t*)ptr;
    ptr += sizeof(int64_t) * ndims;

    indices_ = (int64_t*)ptr;
    ptr += sizeof(int64_t) * nindices;

    values_ = (T*)ptr;
  }

  uint32_t* type_;
  size_t* ndims_;
  int64_t* shape_;
  int64_t* nindices_;
  int64_t* indices_;
  T* values_;
  std::vector<uint8_t> content_;
};