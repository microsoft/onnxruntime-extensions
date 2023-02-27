// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "string_utils.h"
#include "ustring.h"
#include "string_tensor.h"

void GetTensorMutableDataString(const OrtApi& api, OrtW::CustomOpApi& ort, OrtKernelContext* context,
                                const OrtValue* value, std::vector<std::string>& output) {
  (void)context;
  OrtTensorDimensions dimensions(ort, value);
  size_t len = static_cast<size_t>(dimensions.Size());
  size_t data_len;
  OrtW::ThrowOnError(api, api.GetStringTensorDataLength(value, &data_len));
  output.resize(len);
  std::vector<char> result(data_len + len + 1, '\0');
  std::vector<size_t> offsets(len);
  OrtW::ThrowOnError(api, api.GetStringTensorContent(value, (void*)result.data(), data_len, offsets.data(), offsets.size()));
  output.resize(len);
  for (int64_t i = (int64_t)len - 1; i >= 0; --i) {
    if (i < static_cast<int64_t>(len) - 1)
      result[offsets[static_cast<size_t>(i + (int64_t)1)]] = '\0';
    output[static_cast<size_t>(i)] = result.data() + offsets[static_cast<size_t>(i)];
  }
}

void FillTensorDataString(const OrtApi& api, OrtW::CustomOpApi& ort, OrtKernelContext* context,
                          const std::vector<std::string>& value, OrtValue* output) {
  (void)ort;
  (void)context;
  std::vector<const char*> temp(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    temp[i] = value[i].c_str();
  }

  OrtW::ThrowOnError(api,api.FillStringTensor(output, temp.data(), value.size()));
}

void GetTensorMutableDataString(const OrtApi& api, OrtW::CustomOpApi& ort, OrtKernelContext* context,
                                 const OrtValue* value, std::vector<ustring>& output) {
  std::vector<std::string> utf8_strings;
  GetTensorMutableDataString(api, ort, context, value, utf8_strings);

  output.reserve(utf8_strings.size());
  for (auto& str : utf8_strings) {
    output.emplace_back(str);
  }
}


void FillTensorDataString(const OrtApi& api, OrtW::CustomOpApi& ort, OrtKernelContext* context,
                          const std::vector<ustring>& value, OrtValue* output) {
  std::vector<std::string> utf8_strings;
  utf8_strings.reserve(value.size());
  for (const auto& str: value) {
    utf8_strings.push_back(std::string(str));
  }
  FillTensorDataString(api, ort, context, utf8_strings, output);
}
