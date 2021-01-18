// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels.h"

void GetTensorMutableDataString(const OrtApi& api, const OrtValue* value, std::vector<const char*>& output, std::string& result) {
  size_t len, data_len;
  Ort::ThrowOnError(api, api.GetValueCount(value, &len));
  Ort::ThrowOnError(api, api.GetStringTensorDataLength(value, &data_len));
  result.resize(data_len, '\0');
  std::vector<size_t> offsets(len);
  Ort::ThrowOnError(api, api.GetStringTensorContent(value, (void*)result.data(), data_len, offsets.data(), offsets.size()));
  output.resize(len);
  for (size_t i = 0; i < len; ++i) {
    output[i] = result.data() + offsets[i];
  }
}
