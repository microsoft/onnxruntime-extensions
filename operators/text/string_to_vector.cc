#include <charconv>
#include "farmhash.h"
#include "string_utils.h"
#include "string_to_vector.hpp"
#include "string_tensor.h"

StringToVectorImpl::StringToVectorImpl(std::string& map, std::string& unk) {
  ParseMappingTable(map);
  ParseUnkownValue(unk);
}

std::vector<std::vector<int64_t>> StringToVectorImpl::Compute(const std::vector<std::string>& str_input,
                                                              const std::vector<int64_t>& input_dim,
                                                              std::vector<int64_t>& output_dim) const {
  std::vector<std::vector<int64_t>> result;

  // Set output dimension
  output_dim = input_dim;
  output_dim.push_back(vector_len_);

  std::string key;
  for (size_t i = 0; i < str_input.size(); i++) {
    key = str_input[i];

    auto it = map_.find(key);
    if (it != map_.end()) {
      result.push_back(it->second);
    } else {
      result.push_back(unk_value_);
    }
  }

  return result;
}

void StringToVectorImpl::ParseMappingTable(std::string& map) {
  auto lines = SplitString(map, "\n", true);

  if (lines.empty()) {
    return;
  }

  vector_len_ = ParseVectorLen(lines[0]);
  if (vector_len_ == 0) {
    ORTX_CXX_API_THROW(MakeString("The mapped value of string input cannot be empty: ", lines[0]),
                       ORT_INVALID_ARGUMENT);
  }

  std::vector<int64_t> values(vector_len_);
  for (auto& line : lines) {
    auto kv = SplitString(line, "\t", true);

    if (kv.size() != 2) {
      ORTX_CXX_API_THROW(MakeString("Failed to parse mapping_table when processing the line: ", line),
                         ORT_INVALID_ARGUMENT);
    }

    ParseValues(kv[1], values);

    // string to vector mapping
    map_[std::string{kv[0]}] = values;
  }
}

void StringToVectorImpl::ParseUnkownValue(std::string& unk) {
  auto unk_strs = SplitString(unk, " ", true);
  if (unk_strs.size() != vector_len_) {
    ORTX_CXX_API_THROW(
        MakeString("Incompatible dimension: required vector length of unknown_value should be: ", vector_len_),
        ORT_INVALID_ARGUMENT);
  }

  for (auto& str : unk_strs) {
    int64_t value;
    auto [end, ec] = std::from_chars(str.data(), str.data() + str.size(), value);
    if (end != str.data() + str.size()) {
      ORTX_CXX_API_THROW(MakeString("Failed to parse unknown_value when processing the number: ", str),
                         ORT_INVALID_ARGUMENT);
    }

    unk_value_.push_back(value);
  }
}

size_t StringToVectorImpl::ParseVectorLen(const std::string_view& line) {
  auto kv = SplitString(line, "\t", true);

  if (kv.size() != 2) {
    ORTX_CXX_API_THROW(MakeString("Failed to parse mapping_table when processing the line: ", line),
                       ORT_INVALID_ARGUMENT);
  }

  auto value_strs = SplitString(kv[1], " ", true);
  return value_strs.size();
}

void StringToVectorImpl::ParseValues(const std::string_view& v, std::vector<int64_t>& values) {
  std::vector<std::string_view> value_strs = SplitString(v, " ", true);

  int64_t value;
  for (size_t i = 0; i < value_strs.size(); i++) {
    auto [end, ec] = std::from_chars(value_strs[i].data(), value_strs[i].data() + value_strs[i].size(), value);
    if (end != value_strs[i].data() + value_strs[i].size()) {
      ORTX_CXX_API_THROW(MakeString("Failed to parse map when processing the number: ", value_strs[i]),
                         ORT_INVALID_ARGUMENT);
    }
    values[i] = value;
  }
}

OrtStatusPtr KernelStringToVector::OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
  std::string map, unk;
  auto status = OrtW::GetOpAttribute(info, "map", map);
  if (!status) {
    status = OrtW::GetOpAttribute(info, "unk", unk);
  }

  if (!status) {
    impl_ = std::make_shared<StringToVectorImpl>(map, unk);
  }

  return status;
}

OrtStatusPtr KernelStringToVector::Compute(const ortc::Tensor<std::string>& input,
                                           ortc::Tensor<int64_t>& out) const {
  // Setup input
  auto& input_data = input.Data();
  // Get output
  std::vector<int64_t> output_dim;
  auto mapping_result = impl_->Compute(input_data, input.Shape(), output_dim);

  auto* output_data = out.Allocate(output_dim);

  // Set output tensor data
  int idx = 0;
  for (auto& res : mapping_result) {
    for (int64_t value : res) {
      output_data[idx] = value;
      idx++;
    }
  }

  return nullptr;
}
