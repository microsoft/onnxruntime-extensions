// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"
#include "shared/lib/image_processor.h"

using namespace ort_extensions;

class OrtxDeleter {
 public:
  void operator()(ImageProcessor* p) const {
    if (p) {
      OrtxDisposeOnly(p);
    }
  }
};

class ProcessorPtr : public std::unique_ptr<ImageProcessor, OrtxDeleter> {
 public:
  ProcessorPtr(const char* def) {
    OrtxProcessor* proc = nullptr;
    err_ = OrtxCreateProcessor(&proc, def);
    if (err_ == kOrtxOK) {
      reset(static_cast<ImageProcessor*>(proc));
    }
  }

  int err_ = kOrtxOK;
};

auto LoadRawImages(const std::initializer_list<const char*>& image_paths) {
  auto raw_images = std::make_unique<ImageRawData[]>(image_paths.size());
  size_t n = 0;
  for (const auto& image_path : image_paths) {
    std::ifstream ifs(image_path, std::ios::binary);
    if (!ifs.is_open()) {
      break;
    }

    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    ImageRawData& raw_image = raw_images[n++];
    raw_image.resize(size);
    ifs.read(reinterpret_cast<char*>(raw_image.data()), size);
  }

  return std::make_tuple(std::move(raw_images), n);
}

TEST(MultiModelTest, Test1) {
  const char transform_def[] = R"(
{
  "processor": {
    "name": "image_processing",
    "transforms": [
      {
        "operation": {
          "name": "decode_image",
          "domain": "com.microsoft.extensions",
          "type": "DecodeImage",
          "attrs": {
            "color_space": "BGR"
          }
        }
      },
      {
        "operation": {
          "name": "convert_to_rgb",
          "domain": "com.microsoft.extensions",
          "type": "ConvertRGB"
        }
      },
      {
        "operation": {
          "name": "phi3_image_transform",
          "domain": "com.microsoft.extensions",
          "type": "Phi3ImageTransform",
          "attrs": {
            "num_crops": 16,
            "num_img_tokens": 144
          }
        }
      }
    ]
  }
}
  )";

  auto [input_data, n_data] = LoadRawImages({"data/multimodel/exceltable.png", "data/multimodel/australia.jpg"});

  ProcessorPtr proc(transform_def);
  ortc::Tensor<uint8_t>* pixel_values;
  ortc::Tensor<int64_t>* image_sizes;
  ortc::Tensor<int64_t>* num_img_takens;

  auto [status, r] = proc->PreProcess(ort_extensions::span(input_data.get(), (size_t)n_data),
                                 &pixel_values,
                                 &image_sizes,
                                 &num_img_takens);

  proc->ClearOutputs(&r);
  ASSERT_TRUE(status.IsOk());
}
