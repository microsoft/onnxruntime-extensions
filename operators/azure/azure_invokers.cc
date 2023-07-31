// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define CURL_STATICLIB

#include "http_client.h"
#include "curl/curl.h"
#include "azure_invokers.hpp"
#include <sstream>

#define MIN_SUPPORTED_ORT_VER 14

constexpr const char* kUri = "model_uri";
constexpr const char* kModelName = "model_name";
constexpr const char* kModelVer = "model_version";
constexpr const char* kVerbose = "verbose";
constexpr const char* kBinaryType = "binary_type";

struct StringBuffer {
  StringBuffer() = default;
  ~StringBuffer() = default;
  std::stringstream ss_;
};

// apply the callback only when response is for sure to be a '/0' terminated string
static size_t WriteStringCallback(void* contents, size_t size, size_t nmemb, void* userp) {
  try {
    size_t realsize = size * nmemb;
    auto buffer = reinterpret_cast<struct StringBuffer*>(userp);
    buffer->ss_.write(reinterpret_cast<const char*>(contents), realsize);
    return realsize;
  } catch (...) {
    // exception caught, abort write
    return CURLcode::CURLE_WRITE_ERROR;
  }
}

using CurlWriteCallBack = size_t (*)(void*, size_t, size_t, void*);

class CurlHandler {
 public:
  CurlHandler(CurlWriteCallBack call_back) : curl_(curl_easy_init(), curl_easy_cleanup),
                                             headers_(nullptr, curl_slist_free_all),
                                             from_holder_(from_, curl_formfree) {
    curl_easy_setopt(curl_.get(), CURLOPT_BUFFERSIZE, 102400L);
    curl_easy_setopt(curl_.get(), CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl_.get(), CURLOPT_USERAGENT, "curl/7.83.1");
    curl_easy_setopt(curl_.get(), CURLOPT_MAXREDIRS, 50L);
    curl_easy_setopt(curl_.get(), CURLOPT_FTP_SKIP_PASV_IP, 1L);
    curl_easy_setopt(curl_.get(), CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl_.get(), CURLOPT_WRITEFUNCTION, call_back);
  }
  ~CurlHandler() = default;

  void AddHeader(const char* data) {
    headers_.reset(curl_slist_append(headers_.release(), data));
  }
  template <typename... Args>
  void AddForm(Args... args) {
    curl_formadd(&from_, &last_, args...);
  }
  template <typename T>
  void SetOption(CURLoption opt, T val) {
    curl_easy_setopt(curl_.get(), opt, val);
  }
  CURLcode Perform() {
    SetOption(CURLOPT_HTTPHEADER, headers_.get());
    if (from_) {
      SetOption(CURLOPT_HTTPPOST, from_);
    }
    return curl_easy_perform(curl_.get());
  }

 private:
  std::unique_ptr<CURL, decltype(curl_easy_cleanup)*> curl_;
  std::unique_ptr<curl_slist, decltype(curl_slist_free_all)*> headers_;
  curl_httppost* from_{};
  curl_httppost* last_{};
  std::unique_ptr<curl_httppost, decltype(curl_formfree)*> from_holder_;
};

////////////////////// AzureInvoker //////////////////////

AzureInvoker::AzureInvoker(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  auto ver = GetActiveOrtAPIVersion();
  if (ver < MIN_SUPPORTED_ORT_VER) {
    ORTX_CXX_API_THROW("Azure ops requires ort >= 1.14", ORT_RUNTIME_EXCEPTION);
  }

  model_uri_ = TryToGetAttributeWithDefault<std::string>(kUri, "");
  model_name_ = TryToGetAttributeWithDefault<std::string>(kModelName, "");
  model_ver_ = TryToGetAttributeWithDefault<std::string>(kModelVer, "0");
  verbose_ = TryToGetAttributeWithDefault<std::string>(kVerbose, "0");
  OrtStatusPtr status = {};
  size_t input_count = {};
  status = api_.KernelInfo_GetInputCount(&info_, &input_count);
  if (status) {
    ORTX_CXX_API_THROW("failed to get input count", ORT_RUNTIME_EXCEPTION);
  }

  for (size_t ith_input = 0; ith_input < input_count; ++ith_input) {
    char input_name[1024] = {};
    size_t name_size = 1024;
    status = api_.KernelInfo_GetInputName(&info_, ith_input, input_name, &name_size);
    if (status) {
      ORTX_CXX_API_THROW("failed to get input name", ORT_RUNTIME_EXCEPTION);
    }
    input_names_.push_back(input_name);
  }

  size_t output_count = {};
  status = api_.KernelInfo_GetOutputCount(&info_, &output_count);
  if (status) {
    ORTX_CXX_API_THROW("failed to get output count", ORT_RUNTIME_EXCEPTION);
  }

  for (size_t ith_output = 0; ith_output < output_count; ++ith_output) {
    char output_name[1024] = {};
    size_t name_size = 1024;
    status = api_.KernelInfo_GetOutputName(&info_, ith_output, output_name, &name_size);
    if (status) {
      ORTX_CXX_API_THROW("failed to get output name", ORT_RUNTIME_EXCEPTION);
    }
    output_names_.push_back(output_name);
  }
}

////////////////////// AzureAudioInvoker //////////////////////

AzureAudioInvoker::AzureAudioInvoker(const OrtApi& api, const OrtKernelInfo& info) : AzureInvoker(api, info) {
  binary_type_ = TryToGetAttributeWithDefault<std::string>(kBinaryType, "");
}

void AzureAudioInvoker::Compute(const ortc::Variadic& inputs, ortc::Tensor<std::string>& output) const {
  if (inputs.Size() < 1 ||
      inputs[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    ORTX_CXX_API_THROW("invalid inputs, auto token missing", ORT_RUNTIME_EXCEPTION);
  }

  if (inputs.Size() != input_names_.size()) {
    ORTX_CXX_API_THROW("input count mismatch", ORT_RUNTIME_EXCEPTION);
  }

  if (inputs[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING || "auth_token" != input_names_[0]) {
    ORTX_CXX_API_THROW("first input must be a string of auth token", ORT_INVALID_ARGUMENT);
  }

  std::string auth_token = reinterpret_cast<const char*>(inputs[0]->DataRaw());
  std::string full_auth = std::string{"Authorization: Bearer "} + auth_token;

  StringBuffer string_buffer;
  CurlHandler curl_handler(WriteStringCallback);
  curl_handler.AddHeader(full_auth.c_str());
  curl_handler.AddHeader("Content-Type: multipart/form-data");

  for (size_t ith_input = 1; ith_input < inputs.Size(); ++ith_input) {
    switch (inputs[ith_input]->Type()) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        curl_handler.AddForm(CURLFORM_COPYNAME,
                             input_names_[ith_input].c_str(),
                             CURLFORM_COPYCONTENTS,
                             inputs[ith_input]->DataRaw(),
                             CURLFORM_END);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        curl_handler.AddForm(CURLFORM_COPYNAME,
                             input_names_[ith_input].data(),
                             CURLFORM_BUFFER,
                             "non_exist." + binary_type_,
                             CURLFORM_BUFFERPTR,
                             inputs[ith_input]->DataRaw(),
                             CURLFORM_BUFFERLENGTH,
                             inputs[ith_input]->SizeInBytes(),
                             CURLFORM_END);
        break;
      default:
        ORTX_CXX_API_THROW("input must be either text or binary", ORT_RUNTIME_EXCEPTION);
        break;
    }
  }  // for

  curl_handler.SetOption(CURLOPT_URL, model_uri_.c_str());
  curl_handler.SetOption(CURLOPT_VERBOSE, verbose_);
  curl_handler.SetOption(CURLOPT_WRITEDATA, (void*)&string_buffer);

  auto curl_ret = curl_handler.Perform();
  if (CURLE_OK != curl_ret) {
    ORTX_CXX_API_THROW(curl_easy_strerror(curl_ret), ORT_FAIL);
  }

  output.SetStringOutput(std::vector<std::string>{string_buffer.ss_.str()}, std::vector<int64_t>{1L});
}

////////////////////// AzureTextInvoker //////////////////////

AzureTextInvoker::AzureTextInvoker(const OrtApi& api, const OrtKernelInfo& info) : AzureInvoker(api, info) {
}

void AzureTextInvoker::Compute(std::string_view auth, std::string_view input,
                               ortc::Tensor<std::string>& output) const {
  CurlHandler curl_handler(WriteStringCallback);
  StringBuffer string_buffer;

  std::string full_auth = std::string{"Authorization: Bearer "} + auth.data();
  curl_handler.AddHeader(full_auth.c_str());
  curl_handler.AddHeader("Content-Type: application/json");

  curl_handler.SetOption(CURLOPT_URL, model_uri_.c_str());
  curl_handler.SetOption(CURLOPT_POSTFIELDS, input.data());
  curl_handler.SetOption(CURLOPT_POSTFIELDSIZE_LARGE, input.size());
  curl_handler.SetOption(CURLOPT_VERBOSE, verbose_);
  curl_handler.SetOption(CURLOPT_WRITEDATA, (void*)&string_buffer);

  auto curl_ret = curl_handler.Perform();
  if (CURLE_OK != curl_ret) {
    ORTX_CXX_API_THROW(curl_easy_strerror(curl_ret), ORT_FAIL);
  }

  output.SetStringOutput(std::vector<std::string>{string_buffer.ss_.str()}, std::vector<int64_t>{1L});
}

////////////////////// AzureTritonInvoker //////////////////////

namespace tc = triton::client;

AzureTritonInvoker::AzureTritonInvoker(const OrtApi& api, const OrtKernelInfo& info) : AzureInvoker(api, info) {
  auto err = tc::InferenceServerHttpClient::Create(&triton_client_, model_uri_, verbose_ != "0");
}

std::string MapDataType(ONNXTensorElementDataType onnx_data_type) {
  std::string triton_data_type;
  switch (onnx_data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      triton_data_type = "FP32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      triton_data_type = "UINT8";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      triton_data_type = "INT8";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      triton_data_type = "UINT16";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      triton_data_type = "INT16";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      triton_data_type = "INT32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      triton_data_type = "INT64";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      triton_data_type = "BYTES";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      triton_data_type = "BOOL";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      triton_data_type = "FP16";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      triton_data_type = "FP64";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      triton_data_type = "UINT32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      triton_data_type = "UINT64";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      triton_data_type = "BF16";
      break;
    default:
      break;
  }
  return triton_data_type;
}

int8_t* CreateNonStrTensor(const std::string& data_type,
                           ortc::Variadic& outputs,
                           size_t i,
                           const std::vector<int64_t>& shape) {
  if (data_type == "FP32") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<float>(i, shape));
  } else if (data_type == "UINT8") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<uint8_t>(i, shape));
  } else if (data_type == "INT8") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<int8_t>(i, shape));
  } else if (data_type == "UINT16") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<uint16_t>(i, shape));
  } else if (data_type == "INT16") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<int16_t>(i, shape));
  } else if (data_type == "INT32") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<int32_t>(i, shape));
  } else if (data_type == "UINT32") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<uint32_t>(i, shape));
  } else if (data_type == "INT64") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<int64_t>(i, shape));
  } else if (data_type == "UINT64") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<uint64_t>(i, shape));
  } else if (data_type == "BOOL") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<bool>(i, shape));
  } else if (data_type == "FP64") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<double>(i, shape));
  } else {
    return {};
  }
}

#define CHECK_TRITON_ERR(ret, msg)                                                    \
  if (!ret.IsOk()) {                                                                  \
    return ORTX_CXX_API_THROW("Triton err: " + ret.Message(), ORT_RUNTIME_EXCEPTION); \
  }

void AzureTritonInvoker::Compute(const ortc::Variadic& inputs, ortc::Variadic& outputs) const {
  if (inputs.Size() < 1 ||
      inputs[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    ORTX_CXX_API_THROW("invalid inputs, auto token missing", ORT_RUNTIME_EXCEPTION);
  }

  if (inputs.Size() != input_names_.size()) {
    ORTX_CXX_API_THROW("input count mismatch", ORT_RUNTIME_EXCEPTION);
  }

  auto auth_token = reinterpret_cast<const char*>(inputs[0]->DataRaw());
  std::vector<std::unique_ptr<tc::InferInput>> triton_input_vec;
  std::vector<tc::InferInput*> triton_inputs;
  std::vector<std::unique_ptr<const tc::InferRequestedOutput>> triton_output_vec;
  std::vector<const tc::InferRequestedOutput*> triton_outputs;
  tc::Error err;

  for (size_t ith_input = 1; ith_input < inputs.Size(); ++ith_input) {
    tc::InferInput* triton_input = {};
    std::string triton_data_type = MapDataType(inputs[ith_input]->Type());
    if (triton_data_type.empty()) {
      ORTX_CXX_API_THROW("unknow onnx data type", ORT_RUNTIME_EXCEPTION);
    }

    err = tc::InferInput::Create(&triton_input, input_names_[ith_input], inputs[ith_input]->Shape(), triton_data_type);
    CHECK_TRITON_ERR(err, "failed to create triton input");
    triton_input_vec.emplace_back(triton_input);

    triton_inputs.push_back(triton_input);
    if ("BYTES" == triton_data_type) {
      const auto* string_tensor = reinterpret_cast<const ortc::Tensor<std::string>*>(inputs[ith_input].get());
      triton_input->AppendFromString(string_tensor->Data());
    } else {
        const float* data_raw = reinterpret_cast<const float*>(inputs[ith_input]->DataRaw());
        size_t size_in_bytes = inputs[ith_input]->SizeInBytes();
        err = triton_input->AppendRaw(reinterpret_cast<const uint8_t*>(data_raw), size_in_bytes);
        CHECK_TRITON_ERR(err, "failed to append raw data to input");
    }
  }

  for (size_t ith_output = 0; ith_output < output_names_.size(); ++ith_output) {
    tc::InferRequestedOutput* triton_output = {};
    err = tc::InferRequestedOutput::Create(&triton_output, output_names_[ith_output]);
    CHECK_TRITON_ERR(err, "failed to create triton output");
    triton_output_vec.emplace_back(triton_output);
    triton_outputs.push_back(triton_output);
  }

  std::unique_ptr<tc::InferResult> results_ptr;
  tc::InferResult* results = {};
  tc::InferOptions options(model_name_);
  options.model_version_ = model_ver_;
  options.client_timeout_ = 0;

  tc::Headers http_headers;
  http_headers["Authorization"] = std::string{"Bearer "} + auth_token;

  err = triton_client_->Infer(&results, options, triton_inputs, triton_outputs,
                              http_headers, tc::Parameters(),
                              tc::InferenceServerHttpClient::CompressionType::NONE,  // support compression in config?
                              tc::InferenceServerHttpClient::CompressionType::NONE);

  results_ptr.reset(results);
  CHECK_TRITON_ERR(err, "failed to do triton inference");

  size_t output_index = 0;
  auto iter = output_names_.begin();

  while (iter != output_names_.end()) {
    std::vector<int64_t> shape;
    err = results_ptr->Shape(*iter, &shape);
    CHECK_TRITON_ERR(err, "failed to get output shape");

    std::string type;
    err = results_ptr->Datatype(*iter, &type);
    CHECK_TRITON_ERR(err, "failed to get output type");

    if ("BYTES" == type) {
      std::vector<std::string> output_strings;
      err = results_ptr->StringData(*iter, &output_strings);
      CHECK_TRITON_ERR(err, "failed to get output as string");
      auto& string_tensor = outputs.AllocateStringTensor(output_index);
      string_tensor.SetStringOutput(output_strings, shape);
    } else {
      const uint8_t* raw_data = {};
      size_t raw_size;
      err = results_ptr->RawData(*iter, &raw_data, &raw_size);
      CHECK_TRITON_ERR(err, "failed to get output raw data");
      auto* output_raw = CreateNonStrTensor(type, outputs, output_index, shape);
      memcpy(output_raw, raw_data, raw_size);
    }

    ++output_index;
    ++iter;
  }
}

const std::vector<const OrtCustomOp*>& AzureInvokerLoader() {
  static OrtOpLoader op_loader(CustomAzureStruct("AzureAudioInvoker", AzureAudioInvoker),
                               CustomAzureStruct("AzureTritonInvoker", AzureTritonInvoker),
                               CustomAzureStruct("AzureAudioInvoker", AzureAudioInvoker),
                               CustomAzureStruct("AzureTextInvoker", AzureTextInvoker),
                               CustomCpuStruct("AzureAudioInvoker", AzureAudioInvoker),
                               CustomCpuStruct("AzureTritonInvoker", AzureTritonInvoker),
                               CustomCpuStruct("AzureAudioInvoker", AzureAudioInvoker),
                               CustomCpuStruct("AzureTextInvoker", AzureTextInvoker)
  );
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Azure = AzureInvokerLoader;
