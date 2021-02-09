// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once


#include "kernels.h"
#include "farmhash.h"
#include "utils/string_utils.h"

namespace std {

template <class T>
struct hash<std::vector<T>> {

  size_t operator()(const vector<T>& __vector) const noexcept {
    return util::Hash(reinterpret_cast<const char *>(__vector.data()), __vector.size() * sizeof(T));
  }
};
}  // namespace std

template <typename T>
class VectorToStringImpl{
 public:
  VectorToStringImpl(std::string& map, std::string& unk) : unk_value_(unk) {
    ParseMappingTable(map);
  }

  std::vector<std::string> Compute(const void* input, const OrtTensorDimensions& input_dim, OrtTensorDimensions& output_dim) {
    std::vector<std::string> result;


    const T* ptr = static_cast<const T*>(input);

    if (vector_len_ == 1 && input_dim.size() == 1) {
      // only hit when the key is a scalar and the input is a vector
      output_dim = input_dim;
    } else {
      if (input_dim[input_dim.size() - 1] != vector_len_) {
        throw std::runtime_error(MakeString("Incompatible dimension: required vector length should be ", vector_len_));
      }

      output_dim = input_dim;
      output_dim.pop_back();
    }

    std::vector<T> key(vector_len_);
    for (int i = 0; i < input_dim.Size(); i += vector_len_) {
      //construct key
      for (int j = 0; j < vector_len_; j++) {;
        key[j] = ptr[j];
      }

      auto it = map_.find(key);
      if (it != map_.end()) {
        result.push_back(it->second);
      } else {
        result.push_back(unk_value_);
      }

      ptr = ptr + vector_len_;
    }

    return result;
  }

 private:
  void ParseMappingTable(std::string& map) {


    auto lines = SplitString(map, "\n", true);

    if (lines.empty()) {
      return;
    }

    vector_len_ = ParseVectorLen(lines[0]);

    std::vector<T> values(vector_len_);
    for (auto& line : lines) {

      auto kv = SplitString(line, "\t", true);

      if (kv.size() != 2) {
        throw std::runtime_error(MakeString("Failed to parse mapping_table when processing the line: ", line));
      }

      ParseValues(kv[1], values);

      map_[values] = kv[0];
    }

  }

  size_t ParseVectorLen(const std::string_view& line) {
    auto kv = SplitString(line, "\t", true);

    if (kv.size() != 2) {
      throw std::runtime_error(MakeString("Failed to parse mapping_table when processing the line: ", line));
    }

    auto value_strs = SplitString(kv[1], " ", true);
    return value_strs.size();
  }

  void ParseValues(const std::string_view& v, std::vector<T>& values) {
    std::vector<std::string_view> value_strs = SplitString(v, " ", true);

    T value;
    for (int i = 0; i < value_strs.size(); i++) {
      auto [end, ec] = std::from_chars(value_strs[i].data(), value_strs[i].data() + value_strs[i].size(), value);
      if(end != value_strs[i].data() + value_strs[i].size()) {
        throw std::runtime_error(MakeString("Failed to parse map when processing the number: ", value_strs[i]));
      }
      values[i] = value;
    }

  }

  std::unordered_map<std::vector<T>, std::string> map_;
  std::string unk_value_;
  size_t vector_len_;
};

struct KernelVectorToString : BaseKernel {
  KernelVectorToString(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 private:
  std::shared_ptr<VectorToStringImpl<int64_t>> impl_;
};

struct CustomOpVectorToString : Ort::CustomOpBase<CustomOpVectorToString, KernelVectorToString> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};
