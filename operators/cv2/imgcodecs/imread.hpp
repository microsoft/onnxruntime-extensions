#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "status.h"
#include "string_tensor.h"

inline OrtxStatus image_reader(const ortc::Tensor<std::string>& input,
                        ortc::Tensor<uint8_t>& output) {
  auto& input_data_dimensions = input.Shape();
  auto n = input_data_dimensions[0];
  if (n != 1) {
    return {kOrtxErrorInvalidArgument, "[ImageReader]: the dimension of input value can only be 1 now."};
  }
  auto& image_paths = input.Data();
  cv::Mat img = cv::imread(image_paths[0], cv::IMREAD_COLOR);
  std::vector<int64_t> output_dimensions = {1, img.size[0], img.size[1], static_cast<int64_t>(img.elemSize())};
  std::uint8_t* p_output_image = output.Allocate(output_dimensions);
  memcpy(p_output_image, img.data, img.total() * img.elemSize());

  return {};
}
