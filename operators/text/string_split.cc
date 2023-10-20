// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_functions.h"
#include "string_tensor.h"

OrtStatusPtr string_split(const ortc::Tensor<std::string>& input_X,
                  std::string_view sep,
                  bool skip_empty,
                  ortc::Tensor<int64_t>& out_indices,
                  ortc::Tensor<std::string>& out_text,
                  ortc::Tensor<int64_t>& out_shape) {
  // Setup inputs
  auto& X = input_X.Data();

  OrtStatusPtr status = nullptr;
  // Setup output
  auto& dimensions = input_X.Shape();
  if (dimensions.size() != 1) {
    status = OrtW::CreateStatus("Only 1D tensor are supported as input.", ORT_INVALID_ARGUMENT);
    return status;
  }

  std::vector<std::string> words;
  std::vector<int64_t> indices;
  int64_t maxc = 0;
  int64_t col;
  if (sep.size() == 0) {
    char word[2] = "a";
    for (int64_t row = 0; row < dimensions[0]; ++row) {
      const std::string& str = X[static_cast<size_t>(row)];
      if (str.empty())
        continue;
      maxc = static_cast<int64_t>(static_cast<int64_t>(str.size()) > maxc ? str.size() : maxc);
      for (auto it = str.begin(); it != str.end(); ++it) {
        word[0] = *it;
        words.push_back(word);
        indices.push_back(row);
        indices.push_back(std::distance(str.begin(), it));
      }
    }
  } else {
    bool keep = !skip_empty;
    std::size_t current, previous = 0;
    for (int64_t row = 0; row < dimensions[0]; ++row) {
      const std::string& str = X[static_cast<size_t>(row)];
      if (str.empty())
        continue;
      previous = 0;
      col = 0;
      current = str.find_first_of(sep);
      while (current != std::string::npos) {
        if (keep || current > previous) {
          words.push_back(str.substr(previous, current - previous));
          indices.push_back(row);
          indices.push_back(col);
          ++col;
        }
        previous = current + sep.size();
        current = str.find_first_of(sep, previous);
      }
      current = str.size();
      if (keep || current > previous) {
        words.push_back(str.substr(previous, current - previous));
        indices.push_back(row);
        indices.push_back(col);
        ++col;
      }
      maxc = col > maxc ? col : maxc;
    }
  }

  std::vector<int64_t> shape_indices = {static_cast<int64_t>(indices.size()) / 2, 2};
  int64_t* p_indices = out_indices.Allocate(shape_indices);
  std::vector<int64_t> shape_text(1, words.size());
  std::vector<int64_t> shape_shape(1, 2);
  int64_t* p_shape = out_shape.Allocate(shape_shape);

  memcpy(p_indices, indices.data(), indices.size() * sizeof(int64_t));
  p_shape[0] = dimensions[0];
  p_shape[1] = maxc;
  out_text.SetStringOutput(words, shape_text);
  return status;
}
