// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <vector>
#include <string>
#include "string_utils.h"
#include "string_tensor.h"

template <typename T1, typename T2, typename T3>
class BroadcastIteratorRight {
 public:
  BroadcastIteratorRight(const std::vector<int64_t>& shape1,
                         const std::vector<int64_t>& shape2,
                         const T1* p1, const T2* p2, T3* p3) : p1_(p1), p2_(p2), p3_(p3), shape1_(shape1) {
    if (shape2.size() > shape1.size())
      ORT_CXX_API_THROW("shape2 must have less dimensions than shape1", ORT_INVALID_ARGUMENT);
    shape2_.resize(shape1_.size());
    cum_shape2_.resize(shape1_.size());
    total_ = 1;
    for (size_t i = 0; i < shape1_.size(); ++i) {
      total_ *= shape1[i];
      if (i >= shape2.size()) {
        shape2_[i] = 1;
        continue;
      } else {
        shape2_[i] = shape2[i];
      }
      if (shape2[i] != 1 && shape1[i] != shape2[i]) {
        ORT_CXX_API_THROW(MakeString(
            "Cannot broadcast dimension ", i, " left:", shape1[i], " right:", shape2[i]), ORT_INVALID_ARGUMENT);
      }
    }
    cum_shape2_[shape2_.size() - 1] = 1;
    for (size_t i = 1; i < shape1_.size(); ++i) {
      cum_shape2_[shape1_.size() - i - 1] = cum_shape2_[shape1_.size() - i] * shape2_[shape1_.size() - i];
    }
  }

  struct BroadcastIteratorRightState {
    const BroadcastIteratorRight<T1, T2, T3>* parent;
    std::vector<int64_t> index1;
    const T1* p1;
    const T1* end_;
    const T2* p2;
    T3* p3;
    size_t last;
    int dim;

    void init(const BroadcastIteratorRight<T1, T2, T3>& p) {
      parent = &p;
      p1 = p.p1_;
      p2 = p.p2_;
      p3 = p.p3_;
      end_ = p.p1_ + p.total_;
      index1.resize(p.shape1_.size(), 0);
      last = index1.size() - 1;
    }

    bool end() {
      return p1 == end_;
    }

    void next() {
      ++index1[last];
      ++p1;
      ++p3;
      if (parent->shape2_[last] != 1) {
        ++p2;
      }
      dim = static_cast<int>(last);
      while (dim > 0 && index1[dim] >= parent->shape1_[dim]) {
        index1[dim] = 0;
        if (parent->shape2_[dim] != 1) {
          p2 -= parent->cum_shape2_[dim] * parent->shape2_[dim];
        }
        --dim;
        ++index1[dim];
        if (parent->shape2_[dim] != 1) {
          p2 += parent->cum_shape2_[dim];
        }
      }
    }

    template <typename TCMP>
    void loop(TCMP& cmp, BroadcastIteratorRightState& it, int64_t pos = 0) {
      if (pos != 0)
        ORT_CXX_API_THROW("Not implemented yet.", ORT_NOT_IMPLEMENTED);
      while (!end()) {
        *p3 = cmp(*p1, *p2);
        next();
      }
    }
  };

 protected:
  std::vector<int64_t> shape1_;
  std::vector<int64_t> shape2_;
  std::vector<int64_t> cum_shape2_;
  int64_t total_;
  const T1* p1_;
  const T2* p2_;
  T3* p3_;
};

template <typename T>
class Compare {
 public:
  inline bool operator()(const T& s1, const T& s2) const;
};

template <>
inline bool Compare<std::string>::operator()(const std::string& s1, const std::string& s2) const {
  return s1.compare(s2) == 0;
}

template <typename T>
void KernelEqual_Compute(const OrtApi& api, Ort::CustomOpApi& ort_, OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const T* X = ort_.GetTensorData<T>(input_X);
  const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
  const T* Y = ort_.GetTensorData<T>(input_Y);

  // Setup output
  OrtTensorDimensions dimensions_x(ort_, input_X);
  OrtTensorDimensions dimensions_y(ort_, input_Y);
  Compare<T> cmp;

  typename BroadcastIteratorRight<T, T, bool>::BroadcastIteratorRightState state;
  if (dimensions_x.Size() >= dimensions_y.Size()) {
    OrtValue* v = ort_.KernelContext_GetOutput(context, 0, dimensions_x.data(), dimensions_x.size());
    bool* out = ort_.GetTensorMutableData<bool>(v);
    BroadcastIteratorRight<T, T, bool> iter(dimensions_x, dimensions_y, X, Y, out);
    state.init(iter);
    state.loop(cmp, state);
  } else {
    // Operator Equal is commutative.
    OrtValue* v = ort_.KernelContext_GetOutput(context, 0, dimensions_y.data(), dimensions_y.size());
    bool* out = ort_.GetTensorMutableData<bool>(v);
    BroadcastIteratorRight<T, T, bool> iter(dimensions_y, dimensions_x, Y, X, out);
    state.init(iter);
    state.loop(cmp, state);
  }
}

template <>
void KernelEqual_Compute<std::string>(const OrtApi& api, Ort::CustomOpApi& ort_, OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
  std::vector<std::string> X, Y;
  GetTensorMutableDataString(api, ort_, context, input_X, X);
  GetTensorMutableDataString(api, ort_, context, input_Y, Y);

  // Setup output
  OrtTensorDimensions dimensions_x(ort_, input_X);
  OrtTensorDimensions dimensions_y(ort_, input_Y);
  Compare<std::string> cmp;

  typename BroadcastIteratorRight<std::string, std::string, bool>::BroadcastIteratorRightState state;
  if (dimensions_x.Size() >= dimensions_y.Size()) {
    OrtValue* v = ort_.KernelContext_GetOutput(context, 0, dimensions_x.data(), dimensions_x.size());
    bool* out = ort_.GetTensorMutableData<bool>(v);
    BroadcastIteratorRight<std::string, std::string, bool> iter(
        dimensions_x, dimensions_y, X.data(), Y.data(), out);
    state.init(iter);
    state.loop(cmp, state);
  } else {
    // Operator Equal is commutative.
    OrtValue* v = ort_.KernelContext_GetOutput(context, 0, dimensions_y.data(), dimensions_y.size());
    bool* out = ort_.GetTensorMutableData<bool>(v);
    BroadcastIteratorRight<std::string, std::string, bool> iter(
        dimensions_y, dimensions_x, Y.data(), X.data(), out);
    state.init(iter);
    state.loop(cmp, state);
  }
}
