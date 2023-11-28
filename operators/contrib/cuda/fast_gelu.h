// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"

namespace contrib {

template <typename T>
struct FastGelu {
  OrtStatusPtr OnModelAttach(const OrtApi& /*api*/,
                             const OrtKernelInfo& /*info*/) {
    return nullptr;
  }
  /*
  OrtStatusPtr Compute(const ortc::Tensor<T>& input,
                       std::optional<const ortc::Tensor<T>*> bias,
                       ortc::Tensor<T>& output) const;
  */
  OrtStatusPtr Compute(const ortc::Tensor<T>& input,
                       std::optional<const ortc::Tensor<T>*> bias,
                       ortc::Tensor<T>& output) const {
    const T* input_data = input.Data();
    T* output_data = output.Allocate(input.Shape());
    auto input_length = input.NumberOfElement();
    if (0 == input_length) {
        return nullptr;
    }
    const T* bias_data = bias.has_value()?(*bias)->Data():nullptr;
    auto bias_length = bias.has_value()?(*bias)->NumberOfElement():0;
    return nullptr;
  }

 private:
  bool use_half2_ = false; // to-do, read this from env var
};

}