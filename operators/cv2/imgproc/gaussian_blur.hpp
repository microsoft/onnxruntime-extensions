#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct KernelGaussianBlur : BaseKernel {
  KernelGaussianBlur(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  }

  void Compute(OrtKernelContext* context) {
    size_t input_c = ort_.KernelContext_GetInputCount(context);
    const OrtValue* input_data = ort_.KernelContext_GetInput(context, 0);
    const float* p_input_data = ort_.GetTensorData<float>(input_data);
    std::int64_t ksize[] = {3, 3};
    double sigma[] = {0., 0.};
    if (input_c > 1) {
      const OrtValue* input_ksize = ort_.KernelContext_GetInput(context, 1);
      OrtTensorDimensions dim_ksize(ort_, input_ksize);
      if (dim_ksize.size() != 1 || dim_ksize[0] != 2) {
        ORTX_CXX_API_THROW("[GaussianBlur]: ksize shape is (2,)", ORT_INVALID_ARGUMENT);
      }
      std::copy_n(ort_.GetTensorData<std::int64_t>(input_ksize), 2, ksize);
    }

    if (input_c > 2) {
      const OrtValue* input_sigma = ort_.KernelContext_GetInput(context, 2);
      OrtTensorDimensions dim_sigma(ort_, input_sigma);
      if (dim_sigma.size() != 1 || dim_sigma[0] != 2) {
        ORTX_CXX_API_THROW("[GaussianBlur]: sigma shape is (2,)", ORT_INVALID_ARGUMENT);
      }
      std::copy_n(ort_.GetTensorData<double>(input_sigma), 2, sigma);
    }

    OrtTensorDimensions input_data_dimensions(ort_, input_data);

    int n = static_cast<int>(input_data_dimensions[0]);
    int h = static_cast<int>(input_data_dimensions[1]);
    int w = static_cast<int>(input_data_dimensions[2]);
    int c = static_cast<int>(input_data_dimensions[3]);
    (void)n;
    (void)c;

    cv::Mat input_image(cv::Size(w, h), CV_32FC3, (void*)p_input_data);
    cv::Mat output_image;
    cv::GaussianBlur(input_image,
                     output_image,
                     cv::Size(static_cast<int>(ksize[1]), static_cast<int>(ksize[0])),
                     sigma[0], sigma[1], cv::BORDER_DEFAULT);

    OrtValue* image_y = ort_.KernelContext_GetOutput(context,
                                                     0, input_data_dimensions.data(), input_data_dimensions.size());
    float* p_output_image = ort_.GetTensorMutableData<float>(image_y);
    memcpy(p_output_image, output_image.data, output_image.total() * output_image.elemSize());
  }
};

struct CustomOpGaussianBlur : OrtW::CustomOpBase<CustomOpGaussianBlur, KernelGaussianBlur> {
  size_t GetInputTypeCount() const {
    return 3;
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 0) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else if (index == 1) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    } else {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }
  }

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  const char* GetName() const {
    return "GaussianBlur";
  }
};
