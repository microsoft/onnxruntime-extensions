#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

void gaussian_blur(const ortc::TensorT<float>& input_data,
                   const ortc::Span<int64_t>& input_ksize,
                   const ortc::Span<double>& input_sigma,
                   ortc::TensorT<float>& output) {
  const float* p_input_data = input_data.Data();
  std::int64_t ksize[] = {3, 3};
  double sigma[] = {0., 0.};

  if (input_ksize.size() != 2) {
    ORTX_CXX_API_THROW("[GaussianBlur]: ksize shape is (2,)", ORT_INVALID_ARGUMENT);
  }
  std::copy_n(input_ksize.Data(), 2, ksize);

  if (input_sigma.size() != 2) {
    ORTX_CXX_API_THROW("[GaussianBlur]: sigma shape is (2,)", ORT_INVALID_ARGUMENT);
  }
  std::copy_n(input_sigma.Data(), 2, sigma);

  auto& input_data_dimensions = input_data.Shape();

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

  float* p_output_image = output.Allocate(input_data_dimensions);
  memcpy(p_output_image, output_image.data, output_image.total() * output_image.elemSize());
}
