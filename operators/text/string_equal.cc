// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_functions.h"
#include "string_tensor.h"

template <typename T1, typename T2, typename T3>
class BroadcastIteratorRight {
 public:
  BroadcastIteratorRight() = default;
  OrtStatusPtr Init(const std::vector<int64_t>& shape1,
                    const std::vector<int64_t>& shape2,
                    const T1* p1, const T2* p2, T3* p3) {
    shape1_ = shape1;
    p1_ = p1;
    p2_ = p2;
    p3_ = p3;
    if (shape2.size() > shape1.size()) {
      return OrtW::CreateStatus("shape2 must have less dimensions than shape1", ORT_INVALID_ARGUMENT);
    }
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
        return OrtW::CreateStatus(MakeString(
                                      "Cannot broadcast dimension ", i, " left:", shape1[i], " right:", shape2[i])
                                      .c_str(),
                                  ORT_INVALID_ARGUMENT);
      }
    }
    cum_shape2_[shape2_.size() - 1] = 1;
    for (size_t i = 1; i < shape1_.size(); ++i) {
      cum_shape2_[shape1_.size() - i - 1] = cum_shape2_[shape1_.size() - i] * shape2_[shape1_.size() - i];
    }

    return nullptr;
  }

  struct BroadcastIteratorRightState {
    const BroadcastIteratorRight<T1, T2, T3>* parent = nullptr;
    std::vector<int64_t> index1;
    const T1* p1 = nullptr;
    const T1* end_ = nullptr;
    const T2* p2 = nullptr;
    T3* p3 = nullptr;
    size_t last = 0;
    int dim = -1;

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
    OrtStatusPtr loop(TCMP& cmp, BroadcastIteratorRightState& /*it*/, int64_t pos = 0) {
      if (pos != 0)
        return OrtW::CreateStatus("Not implemented yet.", ORT_NOT_IMPLEMENTED);
      while (!end()) {
        *p3 = cmp(*p1, *p2);
        next();
      }
      return nullptr;
    }
  };

 protected:
  OrtStatusPtr status_{};
  std::vector<int64_t> shape1_;
  std::vector<int64_t> shape2_;
  std::vector<int64_t> cum_shape2_;
  int64_t total_{};
  const T1* p1_{};
  const T2* p2_{};
  T3* p3_{};
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

OrtStatusPtr string_equal(const ortc::Tensor<std::string>& input_1,
                          const ortc::Tensor<std::string>& input_2,
                          ortc::Tensor<bool>& output) {
  // Setup inputs
  std::vector<std::string> X = input_1.Data();
  std::vector<std::string> Y = input_2.Data();

  OrtStatusPtr status = nullptr;
  Compare<std::string> cmp;
  typename BroadcastIteratorRight<std::string, std::string, bool>::BroadcastIteratorRightState state;
  if (input_1.NumberOfElement() >= input_2.NumberOfElement()) {
    bool* out = output.Allocate(input_1.Shape());
    BroadcastIteratorRight<std::string, std::string, bool> iter;
    status = iter.Init(
        input_1.Shape(), input_2.Shape(), X.data(), Y.data(), out);
    if (!status) {
      state.init(iter);
      status = state.loop(cmp, state);
    }
  } else {
    // Operator Equal is commutative.
    bool* out = output.Allocate(input_2.Shape());
    BroadcastIteratorRight<std::string, std::string, bool> iter;
    status = iter.Init(
        input_2.Shape(), input_1.Shape(), Y.data(), X.data(), out);
    if (!status) {
      state.init(iter);
      status = state.loop(cmp, state);
    }
  }

  return status;
}
