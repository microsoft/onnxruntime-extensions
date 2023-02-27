#include <charconv>
#include "farmhash.h"
#include "string_utils.h"
#include "string_to_vector.hpp"
#include "string_tensor.h"

StringToVectorImpl::StringToVectorImpl(std::string& map, std::string& unk) {
  ParseMappingTable(map);
  ParseUnkownValue(unk);
}

std::vector<std::vector<int64_t>> StringToVectorImpl::Compute(std::vector<std::string>& str_input, const OrtTensorDimensions& input_dim, OrtTensorDimensions& output_dim) {
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
    ORTX_CXX_API_THROW(MakeString("The mapped value of string input cannot be empty: ", lines[0]), ORT_INVALID_ARGUMENT);
  }

  std::vector<int64_t> values(vector_len_);
  for (auto& line : lines) {
    auto kv = SplitString(line, "\t", true);

    if (kv.size() != 2) {
      ORTX_CXX_API_THROW(MakeString("Failed to parse mapping_table when processing the line: ", line), ORT_INVALID_ARGUMENT);
    }

    ParseValues(kv[1], values);

    // string to vector mapping
    map_[std::string{kv[0]}] = values;
  }
}

void StringToVectorImpl::ParseUnkownValue(std::string& unk) {
  auto unk_strs = SplitString(unk, " ", true);
  if (unk_strs.size() != vector_len_) {
    ORTX_CXX_API_THROW(MakeString("Incompatible dimension: required vector length of unknown_value should be: ", vector_len_), ORT_INVALID_ARGUMENT);
  }

  for (auto& str : unk_strs) {
    int64_t value;
    auto [end, ec] = std::from_chars(str.data(), str.data() + str.size(), value);
    if (end != str.data() + str.size()) {
      ORTX_CXX_API_THROW(MakeString("Failed to parse unknown_value when processing the number: ", str), ORT_INVALID_ARGUMENT);
    }

    unk_value_.push_back(value);
  }
}

size_t StringToVectorImpl::ParseVectorLen(const std::string_view& line) {
  auto kv = SplitString(line, "\t", true);

  if (kv.size() != 2) {
    ORTX_CXX_API_THROW(MakeString("Failed to parse mapping_table when processing the line: ", line), ORT_INVALID_ARGUMENT);
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
      ORTX_CXX_API_THROW(MakeString("Failed to parse map when processing the number: ", value_strs[i]), ORT_INVALID_ARGUMENT);
    }
    values[i] = value;
  }
}

KernelStringToVector::KernelStringToVector(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  std::string map = ort_.KernelInfoGetAttribute<std::string>(&info, "map");
  // unk_value is string here because KernelInfoGetAttribute doesn't support returning vector
  std::string unk = ort_.KernelInfoGetAttribute<std::string>(&info, "unk");

  impl_ = std::make_shared<StringToVectorImpl>(map, unk);
}

void KernelStringToVector::Compute(OrtKernelContext* context) {
  // Setup input
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> input_data;
  GetTensorMutableDataString(api_, ort_, context, input, input_data);
  OrtTensorDimensions input_dim(ort_, input);

  // Get output
  OrtTensorDimensions output_dim;
  auto mapping_result = impl_->Compute(input_data, input_dim, output_dim);

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dim.data(), output_dim.size());
  auto* output_data = ort_.GetTensorMutableData<int64_t>(output);

  // Set output tensor data
  int idx = 0;
  for (auto& res : mapping_result) {
    for (int64_t value : res) {
      output_data[idx] = value;
      idx++;
    }
  }
}

const char* CustomOpStringToVector::GetName() const { return "StringToVector"; };

size_t CustomOpStringToVector::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringToVector::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringToVector::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringToVector::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};
