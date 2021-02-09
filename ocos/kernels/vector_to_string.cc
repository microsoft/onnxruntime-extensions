#include <charconv>
#include "kernels.h"
#include "utils/string_utils.h"
#include "vector_to_string.hpp"
#include "string_common.h"

VectorToStringImpl::VectorToStringImpl(std::string& map, std::string& unk) : unk_value_(unk) {
  ParseMappingTable(map);
}

std::vector<std::string> VectorToStringImpl::Compute(const void* input, const OrtTensorDimensions& input_dim, OrtTensorDimensions& output_dim) {
  std::vector<std::string> result;

  const int64_t* ptr = static_cast<const int64_t*>(input);

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

  std::vector<int64_t> key(vector_len_);
  for (int i = 0; i < input_dim.Size(); i += vector_len_) {
    //construct key
    for (int j = 0; j < vector_len_; j++) {
      ;
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

void VectorToStringImpl::ParseMappingTable(std::string& map) {
  auto lines = SplitString(map, "\n", true);

  if (lines.empty()) {
    return;
  }

  vector_len_ = ParseVectorLen(lines[0]);

  std::vector<int64_t> values(vector_len_);
  for (auto& line : lines) {
    auto kv = SplitString(line, "\t", true);

    if (kv.size() != 2) {
      throw std::runtime_error(MakeString("Failed to parse mapping_table when processing the line: ", line));
    }

    ParseValues(kv[1], values);

    map_[values] = kv[0];
  }
}

size_t VectorToStringImpl::ParseVectorLen(const std::string_view& line) {
  auto kv = SplitString(line, "\t", true);

  if (kv.size() != 2) {
    throw std::runtime_error(MakeString("Failed to parse mapping_table when processing the line: ", line));
  }

  auto value_strs = SplitString(kv[1], " ", true);
  return value_strs.size();
}

void VectorToStringImpl::ParseValues(const std::string_view& v, std::vector<int64_t>& values) {
  std::vector<std::string_view> value_strs = SplitString(v, " ", true);

  int64_t value;
  for (int i = 0; i < value_strs.size(); i++) {
    auto [end, ec] = std::from_chars(value_strs[i].data(), value_strs[i].data() + value_strs[i].size(), value);
    if (end != value_strs[i].data() + value_strs[i].size()) {
      throw std::runtime_error(MakeString("Failed to parse map when processing the number: ", value_strs[i]));
    }
    values[i] = value;
  }
}

KernelVectorToString::KernelVectorToString(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api, info) {
  std::string map = ort_.KernelInfoGetAttribute<std::string>(info, "map");
  std::string unk = ort_.KernelInfoGetAttribute<std::string>(info, "unk");

  // TODO: support more type when we can get input type from OrtKernelInfo
  impl_ = std::make_shared<VectorToStringImpl>(map, unk);
}

void KernelVectorToString::Compute(OrtKernelContext* context) {
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  const void* input_data = ort_.GetTensorData<int64_t>(input);

  OrtTensorDimensions input_dim(ort_, input);
  OrtTensorDimensions output_dim;
  std::vector<std::string> mapping_result = impl_->Compute(input_data, input_dim, output_dim);

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dim.data(), output_dim.size());

  FillTensorDataString(api_, ort_, context, mapping_result, output);
}

void* CustomOpVectorToString::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelVectorToString(api, info);
};

const char* CustomOpVectorToString::GetName() const { return "VectorToString"; };

size_t CustomOpVectorToString::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpVectorToString::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

size_t CustomOpVectorToString::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpVectorToString::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};
