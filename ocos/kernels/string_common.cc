// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_common.h"

void GetTensorMutableDataString(Ort::CustomOpApi& ort, OrtKernelContext* context,
                                const OrtValue* value, std::vector<std::string>& output) {
  OrtTensorDimensions dimensions(ort, value);
  size_t len = static_cast<size_t>(dimensions.Size());
  size_t data_len;
  Ort::ThrowOnError(ort.Api(), ort.Api().GetStringTensorDataLength(value, &data_len));
  output.resize(len);
  std::vector<char> result(data_len + len + 1, '\0');
  std::vector<size_t> offsets(len);
  Ort::ThrowOnError(ort.Api(), ort.Api().GetStringTensorContent(value, (void*)result.data(), data_len, offsets.data(), offsets.size()));
  output.resize(len);
  for (int64_t i = (int64_t)len - 1; i >= 0; --i) {
    if (i < len - 1)
      result[offsets[i + (int64_t)1]] = '\0';
    output[i] = result.data() + offsets[i];
  }
}

void FillTensorDataString(Ort::CustomOpApi& ort, OrtKernelContext* context,
                          const std::vector<std::string>& value, OrtValue* output) {
  std::vector<const char*> temp(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    temp[i] = value[i].c_str();
  }
  ort.Api().FillStringTensor(output, temp.data(), value.size());
}
