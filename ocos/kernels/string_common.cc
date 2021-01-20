// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_common.h"
#include "utils.h"

const OrtApi* _GetApi(Ort::CustomOpApi& ort) {
  if (sizeof(Ort::CustomOpApi) == sizeof(OrtApi*)) {
    // The following line should be replaced when an accessor is available.
    // ort.api_ is not accessible (marked as private)
    // Ort::GetApi() returns null.
    // InitApi is missing when linking.
    // OrtGetApiBase() is missing when linking.
    const OrtApi* api = (const OrtApi*)((OrtApi**)&ort) - 1;
    /*
    // The following code checks api is equal to the expected
    // pointer stored ins Ort::CustomOpApi ort as a private member.
    // The following method can be added `const OrtApi& Api() { return api_; }`
    // to give access to this value.
    if (api != &(ort.Api())) {
      // Ort::InitApi(); - missing from link
      auto diff = (int64_t)(&(ort.Api()) - api);
      throw std::runtime_error(MakeString(
          "Internal error, pointers are different: ",
          api, "!=", &(ort.Api()), " (expected) (other value ", &(Ort::GetApi()),
          ", delta=", diff, ")."));
    }
    */
    return api;
  }
  throw std::runtime_error(MakeString(
      "Unable to get an OrtApi pointer from CustomOpApi. Variable ort is not a pointer. Size ",
      sizeof(Ort::CustomOpApi), "!=", sizeof(OrtApi*), " (expected)."));
}

void GetTensorMutableDataString(Ort::CustomOpApi& ort, OrtKernelContext* context,
                                const OrtValue* value, std::vector<std::string>& output) {
  const OrtApi& api = *_GetApi(ort);
  OrtTensorDimensions dimensions(ort, value);
  size_t len = static_cast<size_t>(dimensions.Size());
  size_t data_len;
  Ort::ThrowOnError(api, api.GetStringTensorDataLength(value, &data_len));
  output.resize(len);
  std::vector<char> result(data_len + len + 1, '\0');
  std::vector<size_t> offsets(len);
  Ort::ThrowOnError(api, api.GetStringTensorContent(value, (void*)result.data(), data_len, offsets.data(), offsets.size()));
  output.resize(len);
  for (int64_t i = (int64_t)len - 1; i >= 0; --i) {
    if (i < len - 1)
      result[offsets[i + (int64_t)1]] = '\0';
    output[i] = result.data() + offsets[i];
  }
}

void FillTensorDataString(Ort::CustomOpApi& ort, OrtKernelContext* context,
                          const std::vector<std::string>& value, OrtValue* output) {
  const OrtApi& api = *_GetApi(ort);
  std::vector<const char*> temp(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    temp[i] = value[i].c_str();
  }
  api.FillStringTensor(output, temp.data(), value.size());
}
