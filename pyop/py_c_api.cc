// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <thread>

#include "ortx_utils.h"
#include "ortx_tokenizer.h"
#include "ortx_processor.h"
#include "pykernel.h"

namespace py = pybind11;

template <typename T>
int64_t NumOfElement(const T& sp) {
  size_t c = 1;
  for (auto v : sp) {
    c *= v;
  }
  return c;
}

void AddGlobalMethodsCApi(pybind11::module& m) {
  m.def(
      "create_processor",
      [](const char* processor_def_json) {
        OrtxProcessor* processor = nullptr;
        auto err = OrtxCreateProcessor(&processor, processor_def_json);
        if (err != kOrtxOK) {
          throw std::runtime_error(std::string("Failed to create processor") + OrtxGetLastErrorMessage());
        }
        return reinterpret_cast<std::uintptr_t>(processor);
      },
      "Create a processor.");

  m.def(
      "load_images",
      [](const std::vector<std::string>& image_paths) {
        OrtxRawImages* images = nullptr;
        size_t num_images = image_paths.size();
        auto image_ptrs = std::make_unique<const char*[]>(num_images);
        for (size_t i = 0; i < num_images; ++i) {
          image_ptrs[i] = image_paths[i].c_str();
        }

        auto err = OrtxLoadImages(&images, image_ptrs.get(), num_images, nullptr);
        if (err != kOrtxOK) {
          throw std::runtime_error(std::string("Failed to load images") + OrtxGetLastErrorMessage());
        }
        return reinterpret_cast<std::uintptr_t>(images);
      },
      "Load images.");

  m.def(
      "image_pre_process",
      [](std::uintptr_t processor_h, std::uintptr_t images_h) {
        OrtxProcessor* processor = reinterpret_cast<OrtxProcessor*>(processor_h);
        OrtxRawImages* images = reinterpret_cast<OrtxRawImages*>(images_h);
        OrtxTensorResult* result{};
        auto err = OrtxImagePreProcess(processor, images, &result);
        if (err != kOrtxOK) {
          throw std::runtime_error(std::string("Failed to preprocess images") + OrtxGetLastErrorMessage());
        }
        return reinterpret_cast<std::uintptr_t>(result);
      },
      "Preprocess images.");

  m.def(
      "tensor_result_get_at",
      [](std::uintptr_t result_h, size_t index) {
        OrtxTensorResult* result = reinterpret_cast<OrtxTensorResult*>(result_h);
        OrtxTensor* tensor{};
        auto err = OrtxTensorResultGetAt(result, index, &tensor);
        if (err != kOrtxOK) {
          throw std::runtime_error(std::string("Failed to get tensor") + OrtxGetLastErrorMessage());
        }

        extDataType_t tensor_type;

        OrtxGetTensorType(tensor, &tensor_type);
        const int64_t* shape{};
        size_t num_dims;
        const void* data{};
        size_t elem_size = 0;
        if (tensor_type == extDataType_t::kOrtxInt64 || tensor_type == extDataType_t::kOrtxFloat) {
          OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&data), &shape, &num_dims);
          elem_size = 4;
          if (tensor_type == extDataType_t::kOrtxInt64) {
            elem_size = 8;
          }
        } else if (tensor_type == extDataType_t::kOrtxUnknownType) {
          throw std::runtime_error("Failed to get tensor type");
        } else if (tensor_type == extDataType_t::kOrtxUnknownType) {
          throw std::runtime_error("unsupported tensor type");
        }

        std::vector<std::size_t> npy_dims;
        for (auto n = num_dims - num_dims; n < num_dims; ++n) {
          npy_dims.push_back(shape[n]);
        }
        py::array obj{};

        if (tensor_type == extDataType_t::kOrtxFloat) {
          obj = py::array_t<float>(npy_dims);
        } else if (tensor_type == extDataType_t::kOrtxInt64) {
          obj = py::array_t<int64_t>(npy_dims);
        }

        void* out_ptr = obj.mutable_data();
        memcpy(out_ptr, data, NumOfElement(npy_dims) * elem_size);
        return obj;
      },
      "Get tensor at index.");

  m.def(
      "create_tokenizer",
      [](std::string tokenizer_def_json) {
        OrtxTokenizer* tokenizer = nullptr;
        auto err = OrtxCreateTokenizer(&tokenizer, tokenizer_def_json.c_str());
        if (err != kOrtxOK) {
          throw std::runtime_error(std::string("Failed to create tokenizer") + OrtxGetLastErrorMessage());
        }
        return reinterpret_cast<std::uintptr_t>(tokenizer);
      },
      "Create a tokenizer.");

  m.def(
      "batch_tokenize",
      [](std::uintptr_t h, const std::vector<std::string>& inputs) -> std::vector<std::vector<int64_t>> {
        std::vector<std::vector<int64_t>> output;
        OrtxTokenizer* tokenizer = reinterpret_cast<OrtxTokenizer*>(h);
        OrtxTokenId2DArray* tid_output = nullptr;
        std::vector<const char*> cs_inputs;
        for (const auto& input : inputs) {
          cs_inputs.push_back(input.c_str());
        }
        auto err = OrtxTokenize(tokenizer, cs_inputs.data(), inputs.size(), &tid_output);
        if (err != kOrtxOK) {
          throw std::runtime_error(std::string("Failed to tokenize") + OrtxGetLastErrorMessage());
        }

        for (size_t i = 0; i < inputs.size(); ++i) {
          const extTokenId_t* t2d{};
          size_t length{};
          err = OrtxTokenId2DArrayGetItem(tid_output, i, &t2d, &length);
          if (err != kOrtxOK) {
            throw std::runtime_error(std::string("Failed to get token id") + OrtxGetLastErrorMessage());
          }
          output.push_back(std::vector<int64_t>(t2d, t2d + length));
        }
        OrtxDisposeOnly(tid_output);
        return output;
      },
      "Batch tokenize.");

  m.def(
      "batch_detokenize",
      [](std::uintptr_t h, const std::vector<std::vector<int64_t>>& inputs) -> std::vector<std::string> {
        std::vector<std::string> result;
        OrtxTokenizer* tokenizer = reinterpret_cast<OrtxTokenizer*>(h);
        OrtxStringArray* output = nullptr;
        for (size_t i = 0; i < inputs.size(); ++i) {
          std::vector<extTokenId_t> input;
          input.reserve(inputs[i].size());
          std::transform(inputs[i].begin(), inputs[i].end(), std::back_inserter(input),
                         [](int64_t v) { return static_cast<extTokenId_t>(v); });

          auto err = OrtxDetokenize1D(tokenizer, input.data(), input.size(), &output);
          if (err != kOrtxOK) {
            throw std::runtime_error(std::string("Failed to detokenize") + OrtxGetLastErrorMessage());
          }
          size_t length;
          err = OrtxStringArrayGetBatch(output, &length);
          if (err != kOrtxOK) {
            throw std::runtime_error(std::string("Failed to get batch size") + OrtxGetLastErrorMessage());
          }
          for (size_t i = 0; i < length; ++i) {
            const char* item;
            err = OrtxStringArrayGetItem(output, i, &item);
            if (err != kOrtxOK) {
              throw std::runtime_error(std::string("Failed to get item") + OrtxGetLastErrorMessage());
            }
            result.push_back(item);
          }
          OrtxDisposeOnly(output);
        }
        return result;
      },
      "Batch detokenize.");

  m.def(
      "delete_object", [](std::uintptr_t h) { OrtxDisposeOnly(reinterpret_cast<OrtxObject*>(h)); },
      "Delete the object created by C API.");
}
