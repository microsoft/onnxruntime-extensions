#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "string_tensor.h"

struct KernelImageReader : BaseKernel {
  KernelImageReader(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  }

  void Compute(OrtKernelContext* context) {
    const OrtValue* input_data = ort_.KernelContext_GetInput(context, 0);
    OrtTensorDimensions input_data_dimensions(ort_, input_data);

    int n = input_data_dimensions[0];
    if (n != 1) {
      ORTX_CXX_API_THROW("[ImageReader]: the dimension of input value can only be 1 now.", ORT_INVALID_ARGUMENT);
    }

    std::vector<std::string> image_paths;
    GetTensorMutableDataString(api_, ort_, context, input_data, image_paths);

    cv::Mat img = cv::imread(image_paths[0], cv::IMREAD_COLOR);
    std::vector<int64_t> output_dimensions = {1, img.size[0], img.size[1], static_cast<int64_t>(img.elemSize())};
    OrtValue* output_image = ort_.KernelContext_GetOutput(context, 0, output_dimensions.data(), output_dimensions.size());
    std::uint8_t* p_output_image = ort_.GetTensorMutableData<uint8_t>(output_image);
    memcpy(p_output_image, img.data, img.total() * img.elemSize());
  }
};

struct CustomOpImageReader : OrtW::CustomOpBase<CustomOpImageReader, KernelImageReader> {
  size_t GetInputTypeCount() const {
    return 1;
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  }

  const char* GetName() const {
    return "ImageReader";
  }
};
