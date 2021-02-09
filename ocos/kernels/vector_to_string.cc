#include <memory>
#include <unordered_map>
#include <vector>
#include <charconv>
#include "kernels.h"
#include "utils/string_utils.h"
#include "vector_to_string.hpp"
#include "string_common.h"


KernelVectorToString::KernelVectorToString(OrtApi api, const OrtKernelInfo* info) :
  BaseKernel(api, info) {
  std::string map = ort_.KernelInfoGetAttribute<std::string>(info, "map");
  std::string unk = ort_.KernelInfoGetAttribute<std::string>(info, "unk");

  // TODO: support more type when we can get input type from OrtKernelInfo
  impl_ = std::make_shared<VectorToStringImpl<int64_t>>(map, unk);
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
