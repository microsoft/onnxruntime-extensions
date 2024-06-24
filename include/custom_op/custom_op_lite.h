// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <numeric>
#include <string>
#include <string_view>
#include "tensor_api.h"
#include "ort_c_to_cpp.h"
#include "onnxruntime_f16.h"

namespace Ort {
namespace Custom {

class OrtKernelContextStorage : public ITensorStorage {
 public:
  OrtKernelContextStorage(const OrtW::CustomOpApi& custom_op_api,
                          OrtKernelContext& ctx,
                          size_t indice,
                          bool is_input) : api_(custom_op_api), ctx_(ctx), indice_(indice) {
    if (is_input) {
      auto input_count = api_.KernelContext_GetInputCount(&ctx);
      if (indice >= input_count) {
        ORTX_CXX_API_THROW("invalid indice", ORT_RUNTIME_EXCEPTION);
      }
      const_value_ = api_.KernelContext_GetInput(&ctx, indice);
      auto* info = api_.GetTensorTypeAndShape(const_value_);
      shape_ = api_.GetTensorShape(info);
      api_.ReleaseTensorTypeAndShapeInfo(info);
    }
  }

  const std::vector<int64_t>& Shape() const override {
    if (!IsInitialized())
      ORTX_CXX_API_THROW("Tensor not initialized", ORT_RUNTIME_EXCEPTION);
    return *shape_;
  }

  virtual bool IsInitialized() const override {
    return shape_.has_value();
  }

  const void* DataRaw() const override {
    return api_.GetTensorRawData(const_value_);
  }

  void* Initialize(const std::vector<int64_t>& shape, size_t element_size) override {
    if (!const_value_) {
      const_value_ = api_.KernelContext_GetOutput(&ctx_, indice_, shape.data(), shape.size());
      shape_ = shape;
    }
    return api_.GetTensorMutableRawData(const_cast<OrtValue*>(const_value_));
  }

  void* Release() override {
    ORTX_CXX_API_THROW("Can't release the tensor buffer with ORT graph mode.", ORT_RUNTIME_EXCEPTION);
  }

 private:
  const OrtW::CustomOpApi& api_;
  OrtKernelContext& ctx_;
  size_t indice_;
  const OrtValue* const_value_{};  // for input
  std::optional<std::vector<int64_t>> shape_;
};

static std::string get_mem_type(const OrtW::CustomOpApi& custom_op_api,
                                OrtKernelContext& ctx,
                                size_t indice,
                                bool is_input) {
  std::string output = "Cpu";
  if (is_input) {
    const OrtValue* const_value = custom_op_api.KernelContext_GetInput(&ctx, indice);
    const OrtMemoryInfo* mem_info = {};
    custom_op_api.ThrowOnError(custom_op_api.GetOrtApi().GetTensorMemoryInfo(const_value, &mem_info));
    if (mem_info) {
      const char* mem_type = nullptr;
      custom_op_api.ThrowOnError(custom_op_api.GetOrtApi().MemoryInfoGetName(mem_info, &mem_type));
      if (mem_type) {
        output = mem_type;
      }
    }
  }
  return output;
}

template <typename T>
class OrtTensor : public Tensor<T> {
 public:
  OrtTensor(const OrtW::CustomOpApi& custom_op_api,
            OrtKernelContext& ctx,
            size_t indice,
            bool is_input) : Tensor<T>(std::make_unique<OrtKernelContextStorage>(custom_op_api, ctx, indice, is_input)),
                             mem_type_(get_mem_type(custom_op_api, ctx, indice, is_input)) {
  }

  bool IsCpuTensor() const {
    return mem_type_ == "Cpu";
  }

 private:
  std::string mem_type_ = "Cpu";
};

class OrtStringTensorStorage : public IStringTensorStorage<std::string> {
 public:
  using strings = std::vector<std::string>;
  OrtStringTensorStorage(const OrtW::CustomOpApi& custom_op_api,
                         OrtKernelContext& ctx,
                         size_t indice,
                         bool is_input) : api_(custom_op_api), ctx_(ctx), indice_(indice) {
    if (is_input) {
      auto input_count = api_.KernelContext_GetInputCount(&ctx_);
      if (indice >= input_count) {
        ORTX_CXX_API_THROW("invalid indice", ORT_RUNTIME_EXCEPTION);
      }

      auto* const_value = api_.KernelContext_GetInput(&ctx_, indice);
      auto* info = api_.GetTensorTypeAndShape(const_value);
      shape_ = api_.GetTensorShape(info);
      api_.ReleaseTensorTypeAndShapeInfo(info);

      size_t num_chars;
      OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().GetStringTensorDataLength(const_value, &num_chars));
      std::vector<char> chars(num_chars + 1, '\0');
      // assert((*shape_).size() == 1 || ((*shape_).size() == 2 && (*shape_)[0] == 1));

      int64_t num_strings = 1;  // string scalar
      if ((*shape_).size() > 0) {
        num_strings = (*shape_).front();
        for (auto iter = (*shape_).begin() + 1; iter != (*shape_).end(); ++iter) {
          num_strings *= *iter;
        }
      }
      std::vector<size_t> offsets(num_strings);
      OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().GetStringTensorContent(const_value,
                                                                                   (void*)chars.data(),
                                                                                   num_chars,
                                                                                   offsets.data(),
                                                                                   offsets.size()));
      auto upper_bound = static_cast<int64_t>(num_strings) - 1;
      input_strings_.resize(num_strings);
      for (int64_t i = upper_bound; i >= 0; --i) {
        if (i < upper_bound) {
          chars[offsets[i + 1]] = '\0';
        }
        input_strings_[i] = chars.data() + offsets[i];
      }
    }
  }

  const std::vector<int64_t>& Shape() const override {
    if (!IsInitialized())
      ORTX_CXX_API_THROW("Tensor not initialized", ORT_RUNTIME_EXCEPTION);
    return *shape_;
  }

  virtual const void* DataRaw() const override {
    if (input_strings_.size() != 1) {
      ORTX_CXX_API_THROW("DataRaw() only applies to string scalar", ORT_RUNTIME_EXCEPTION);
    }
    return reinterpret_cast<const void*>(input_strings_[0].c_str());
  }

  virtual bool IsInitialized() const override {
    return shape_.has_value();
  }

  virtual void SetStringOutput(const strings& ss, const std::vector<int64_t>& dims) override {
    std::vector<const char*> raw;
    for (const auto& s : ss) {
      raw.push_back(s.data());
    }
    auto* output = api_.KernelContext_GetOutput(&ctx_, indice_, dims.data(), dims.size());
    OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().FillStringTensor(output, raw.data(), raw.size()));
  }

  virtual void SetStringOutput(const std::vector<const char*>& ss, const std::vector<int64_t>& dims) override {
    auto* output = api_.KernelContext_GetOutput(&ctx_, indice_, dims.data(), dims.size());
    OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().FillStringTensor(output, ss.data(), ss.size()));
  }

  const strings& Data() const override {
    return input_strings_;
  }

 private:
  const OrtW::CustomOpApi& api_;
  OrtKernelContext& ctx_;
  size_t indice_;
  std::vector<std::string> input_strings_;
  std::optional<std::vector<int64_t>> shape_;
};

class OrtStringViewTensorStorage : public IStringTensorStorage<std::string_view> {
 public:
  using strings = std::vector<std::string_view>;
  OrtStringViewTensorStorage(const OrtW::CustomOpApi& custom_op_api,
                             OrtKernelContext& ctx,
                             size_t indice,
                             bool is_input) : api_(custom_op_api), ctx_(ctx), indice_(indice) {
    if (is_input) {
      auto input_count = api_.KernelContext_GetInputCount(&ctx_);
      if (indice >= input_count) {
        ORTX_CXX_API_THROW("invalid indice", ORT_RUNTIME_EXCEPTION);
      }
      auto* const_value = api_.KernelContext_GetInput(&ctx_, indice);
      auto* info = api_.GetTensorTypeAndShape(const_value);
      shape_ = api_.GetTensorShape(info);
      api_.ReleaseTensorTypeAndShapeInfo(info);

      size_t num_chars;
      OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().GetStringTensorDataLength(const_value, &num_chars));
      chars_.resize(num_chars + 1, '\0');

      size_t num_strings = 1;
      if ((*shape_).size() > 0) {
        num_strings = static_cast<size_t>((*shape_)[0]);
      }

      if (num_strings) {
        std::vector<size_t> offsets(num_strings);
        OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().GetStringTensorContent(const_value,
                                                                                     (void*)chars_.data(),
                                                                                     num_chars,
                                                                                     offsets.data(),
                                                                                     offsets.size()));
        offsets.push_back(num_chars);
        for (size_t i = 0; i < num_strings; ++i) {
          input_string_views_.emplace_back(chars_.data() + offsets[i], offsets[i + 1] - offsets[i]);
        }
      }
    }
  }

  const std::vector<int64_t>& Shape() const override {
    if (!IsInitialized())
      ORTX_CXX_API_THROW("Tensor not initialized", ORT_RUNTIME_EXCEPTION);
    return *shape_;
  }

  virtual const void* DataRaw() const override {
    if (input_string_views_.size() != 1) {
      ORTX_CXX_API_THROW("DataRaw() only applies to string scalar", ORT_RUNTIME_EXCEPTION);
    }
    return reinterpret_cast<const void*>(input_string_views_[0].data());
  }

  virtual bool IsInitialized() const override {
    return shape_.has_value();
  }

  virtual void SetStringOutput(const strings& ss, const std::vector<int64_t>& dims) override {
    ORTX_CXX_API_THROW("Set output for string view tensor is not supported", ORT_RUNTIME_EXCEPTION);
  }

  virtual void SetStringOutput(const std::vector<const char*>& ss, const std::vector<int64_t>& dims) override {
    ORTX_CXX_API_THROW("Set output for string view tensor is not supported", ORT_RUNTIME_EXCEPTION);
  }

  const strings& Data() const override {
    return input_string_views_;
  }

 private:
  const OrtW::CustomOpApi& api_;
  OrtKernelContext& ctx_;
  size_t indice_;
  std::vector<char> chars_;                           // for input
  std::vector<std::string_view> input_string_views_;  // for input
  std::optional<std::vector<int64_t>> shape_;
};

// to make the metaprogramming magic happy.
template <>
class OrtTensor<std::string> : public Tensor<std::string> {
 public:
  OrtTensor(const OrtW::CustomOpApi& custom_op_api,
            OrtKernelContext& ctx,
            size_t indice,
            bool is_input) : Tensor<std::string>(std::make_unique<OrtStringTensorStorage>(custom_op_api, ctx, indice, is_input)),
                             mem_type_(get_mem_type(custom_op_api, ctx, indice, is_input)) {}

  bool IsCpuTensor() const {
    return mem_type_ == "Cpu";
  }

 private:
  std::string mem_type_ = "Cpu";
};

template <>
class OrtTensor<std::string_view> : public Tensor<std::string_view> {
 public:
  OrtTensor(const OrtW::CustomOpApi& custom_op_api,
            OrtKernelContext& ctx,
            size_t indice,
            bool is_input) : Tensor<std::string_view>(std::make_unique<OrtStringViewTensorStorage>(custom_op_api, ctx, indice, is_input)),
                             mem_type_(get_mem_type(custom_op_api, ctx, indice, is_input)) {}

  bool IsCpuTensor() const {
    return mem_type_ == "Cpu";
  }

 private:
  std::string mem_type_ = "Cpu";
};

using TensorPtr = std::unique_ptr<Custom::Arg>;
using TensorPtrs = std::vector<TensorPtr>;

using TensorBasePtr = std::unique_ptr<Custom::TensorBase>;
using TensorBasePtrs = std::vector<TensorBasePtr>;

// Represent variadic input or output
struct Variadic : public Arg {
  Variadic(const OrtW::CustomOpApi& custom_op_api,
           OrtKernelContext& ctx,
           size_t indice,
           bool is_input) : api_(custom_op_api), ctx_(ctx), indice_(indice), mem_type_(get_mem_type(custom_op_api, ctx, indice, is_input)) {
#if ORT_API_VERSION < 14
    ORTX_CXX_API_THROW("Variadic input or output only supported after onnxruntime 1.14", ORT_RUNTIME_EXCEPTION);
#endif
    if (is_input) {
      auto input_count = api_.KernelContext_GetInputCount(&ctx_);
      for (size_t ith_input = 0; ith_input < input_count; ++ith_input) {
        auto* const_value = api_.KernelContext_GetInput(&ctx_, ith_input);
        auto* info = api_.GetTensorTypeAndShape(const_value);
        auto type = api_.GetTensorElementType(info);
        api_.ReleaseTensorTypeAndShapeInfo(info);
        TensorBasePtr tensor;
        switch (type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            tensor = std::make_unique<Custom::OrtTensor<bool>>(api_, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            tensor = std::make_unique<Custom::OrtTensor<float>>(api_, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            tensor = std::make_unique<Custom::OrtTensor<double>>(api_, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            tensor = std::make_unique<Custom::OrtTensor<uint8_t>>(api_, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            tensor = std::make_unique<Custom::OrtTensor<int8_t>>(api_, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            tensor = std::make_unique<Custom::OrtTensor<uint16_t>>(api_, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            tensor = std::make_unique<Custom::OrtTensor<int16_t>>(api_, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            tensor = std::make_unique<Custom::OrtTensor<uint32_t>>(api_, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            tensor = std::make_unique<Custom::OrtTensor<int32_t>>(api_, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            tensor = std::make_unique<Custom::OrtTensor<uint64_t>>(api_, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            tensor = std::make_unique<Custom::OrtTensor<int64_t>>(api_, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            tensor = std::make_unique<Custom::OrtTensor<std::string>>(api_, ctx, ith_input, true);
            break;
          default:
            ORTX_CXX_API_THROW("unknow input type", ORT_RUNTIME_EXCEPTION);
            break;
        }
        tensors_.emplace_back(tensor.release());
      }  // for
    } else {
      // a Variadic used for output is populated by the Compute so leave tensors_ empty here
    }
  }
  template <typename T>
  T* AllocateOutput(size_t ith_output, const std::vector<int64_t>& shape) {
    auto tensor = std::make_unique<OrtTensor<T>>(api_, ctx_, ith_output, false);
    auto raw_output = tensor.get()->Allocate(shape);
    tensors_.emplace_back(tensor.release());
    return raw_output;
  }
  Tensor<std::string>& AllocateStringTensor(size_t ith_output) {
    auto tensor = std::make_unique<OrtTensor<std::string>>(api_, ctx_, ith_output, false);
    Tensor<std::string>& output = *tensor;
    tensors_.emplace_back(tensor.release());
    return output;
  }
  size_t Size() const {
    return tensors_.size();
  }

  const TensorBasePtr& operator[](size_t indice) const {
    return tensors_.at(indice);
  }

 private:
  const OrtW::CustomOpApi& api_;
  OrtKernelContext& ctx_;
  size_t indice_;
  std::string mem_type_ = "Cpu";
  TensorBasePtrs tensors_;
};

#if ORT_API_VERSION >= 17

class OrtGraphKernelContext : public KernelContext {
 public:
  OrtGraphKernelContext(const OrtApi& ort_api, const OrtKernelContext& ctx) : api_(ort_api) {
    OrtMemoryInfo* info;
    OrtW::ThrowOnError(api_, api_.CreateCpuMemoryInfo(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault, &info));
    OrtW::ThrowOnError(api_, api_.KernelContext_GetAllocator(&ctx, info, &allocator_));
    api_.ReleaseMemoryInfo(info);
  }

  virtual ~OrtGraphKernelContext() {
    if (allocator_) {
      api_.ReleaseAllocator(allocator_);
    }
  }

  void* AllocScratchBuffer(size_t size) override {
    return allocator_->Alloc(allocator_, size);
  }

  void FreeScratchBuffer(void* p) override {
    if (p) {
      allocator_->Free(allocator_, p);
    }
  }

 private:
  const OrtApi& api_;
  OrtAllocator* allocator_;
};

#endif

#ifdef USE_CUDA

enum CudaResource {
  cuda_handle_t = 10000,
  cudnn_handle_t,
  cublas_handle_t,
  deferred_cpu_allocator_t,
  // below are cuda ep options
  device_id_t,
};

#if ORT_API_VERSION >= 17
class OrtGraphCudaKernelContext : public CUDAKernelContext {
 public:
  static const int cuda_resource_ver = 1;

  OrtGraphCudaKernelContext(const OrtApi& ort_api, const OrtKernelContext& ctx) : api_(ort_api) {
    api_.KernelContext_GetResource(&ctx, cuda_resource_ver, CudaResource::cuda_handle_t, &cuda_stream_);
    if (!cuda_stream_) {
      ORTX_CXX_API_THROW("Failed to fetch cuda stream from context", ORT_RUNTIME_EXCEPTION);
    }
    api_.KernelContext_GetResource(&ctx, cuda_resource_ver, CudaResource::cublas_handle_t, &cublas_);
    if (!cublas_) {
      ORTX_CXX_API_THROW("Failed to fetch cublas handle from context", ORT_RUNTIME_EXCEPTION);
    }
    void* resource = nullptr;
    OrtStatusPtr result = api_.KernelContext_GetResource(&ctx, cuda_resource_ver, CudaResource::device_id_t, &resource);
    if (result) {
      ORTX_CXX_API_THROW("Failed to fetch device id from context", ORT_RUNTIME_EXCEPTION);
    }
    memcpy(&device_id_, &resource, sizeof(int));

    OrtMemoryInfo* info;
    OrtW::ThrowOnError(api_, api_.CreateCpuMemoryInfo(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault, &info));
    OrtW::ThrowOnError(api_, api_.KernelContext_GetAllocator(&ctx, info, &cpu_allocator_));
    api_.ReleaseMemoryInfo(info);

    OrtMemoryInfo* cuda_mem_info;
    OrtW::ThrowOnError(api_, api_.CreateMemoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, device_id_, OrtMemType::OrtMemTypeDefault, &cuda_mem_info));
    OrtW::ThrowOnError(api_, api_.KernelContext_GetAllocator(&ctx, cuda_mem_info, &cuda_allocator_));
    api_.ReleaseMemoryInfo(cuda_mem_info);
  }

  virtual ~OrtGraphCudaKernelContext() {
    if (cpu_allocator_) {
      api_.ReleaseAllocator(cpu_allocator_);
    }
    if (cuda_allocator_) {
      api_.ReleaseAllocator(cuda_allocator_);
    }
  }

  void* AllocScratchBuffer(size_t size) override {
    return cpu_allocator_->Alloc(cpu_allocator_, size);
  }

  void FreeScratchBuffer(void* p) override {
    if (p) {
      cpu_allocator_->Free(cpu_allocator_, p);
    }
  }

  void* AllocCudaScratchBuffer(size_t size) override {
    return cuda_allocator_->Alloc(cuda_allocator_, size);
  }

  void FreeCudaScratchBuffer(void* p) override {
    if (p) {
      cuda_allocator_->Free(cuda_allocator_, p);
    }
  }

  void* GetCudaStream() const override {
    return cuda_stream_;
  }

  void* GetCublasHandle() const override {
    return cublas_;
  }

  int GetCudaDeviceId() const override {
    return device_id_;
  }

 private:
  const OrtApi& api_;
  OrtAllocator* cpu_allocator_;
  OrtAllocator* cuda_allocator_;
  void* cuda_stream_ = {};
  void* cublas_ = {};
  int device_id_ = 0;
};

#endif
#endif

// using mf16_t = uint16_t;

struct OrtLiteCustomOp : public OrtCustomOp {
  // CreateTuple
  template <size_t ith_input, size_t ith_output, typename... Ts>
  static typename std::enable_if<sizeof...(Ts) == 0, std::tuple<>>::type
  CreateTuple(const OrtW::CustomOpApi*, OrtKernelContext*, std::vector<TensorPtr>&, size_t, size_t, const std::string&) {
    return std::make_tuple();
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, OrtKernelContext*>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    std::tuple<T> current = std::tuple<OrtKernelContext*>{context};
    auto next = CreateTuple<ith_input, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }

#if ORT_API_VERSION >= 17
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, KernelContext*>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    tensors.push_back(std::make_unique<OrtGraphKernelContext>(api->GetOrtApi(), *context));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(tensors.back().get())};
    auto next = CreateTuple<ith_input, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }

#ifdef USE_CUDA

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, CUDAKernelContext*>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    tensors.push_back(std::make_unique<OrtGraphCudaKernelContext>(api->GetOrtApi(), *context));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(tensors.back().get())};
    auto next = CreateTuple<ith_input, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }

#endif

#endif

#if ORT_API_VERSION >= 14
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Variadic*>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    tensors.push_back(std::make_unique<Variadic>(*api, *context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(tensors.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Variadic&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    tensors.push_back(std::make_unique<Variadic>(*api, *context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Variadic*>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    tensors.push_back(std::make_unique<Variadic>(*api, *context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(tensors.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Variadic&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    tensors.push_back(std::make_unique<Variadic>(*api, *context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }
#endif

#undef data_type_def
#define data_type_def bool
#include "tensor_tuple.inc"

#if ORT_API_VERSION >= 16
#undef data_type_def
#define data_type_def BFloat16
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def MFloat16
#include "tensor_tuple.inc"
#endif

#undef data_type_def
#define data_type_def float
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def double
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def int8_t
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def int16_t
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def int32_t
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def int64_t
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def uint8_t
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def uint16_t
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def uint32_t
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def uint64_t
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def std::string
#include "tensor_tuple.inc"

#undef data_type_def
#define data_type_def std::string_view
#include "tensor_tuple.inc"

#undef data_type_def

  // ParseArgs ...
  template <typename... Ts>
  static typename std::enable_if<0 == sizeof...(Ts)>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>&, std::vector<ONNXTensorElementDataType>&) {
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, OrtKernelContext*>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    ParseArgs<Ts...>(input_types, output_types);
  }

#if ORT_API_VERSION >= 17
  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, KernelContext*>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    ParseArgs<Ts...>(input_types, output_types);
  }

#ifdef USE_CUDA
  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, CUDAKernelContext*>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    ParseArgs<Ts...>(input_types, output_types);
  }
#endif
#endif

#if ORT_API_VERSION >= 14
  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Variadic&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    if (!input_types.empty()) {
      ORTX_CXX_API_THROW("for op has variadic input, only one input is allowed", ORT_RUNTIME_EXCEPTION);
    }
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Variadic*>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    if (!input_types.empty()) {
      ORTX_CXX_API_THROW("for op has variadic input, only one input is allowed", ORT_RUNTIME_EXCEPTION);
    }
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Variadic&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    if (!output_types.empty()) {
      ORTX_CXX_API_THROW("for op has variadic output, only one output is allowed", ORT_RUNTIME_EXCEPTION);
    }
    output_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Variadic*>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    if (!output_types.empty()) {
      ORTX_CXX_API_THROW("for op has variadic output, only one output is allowed", ORT_RUNTIME_EXCEPTION);
    }
    output_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
    ParseArgs<Ts...>(input_types, output_types);
  }
#endif

#define PARSE_INPUT_BASE(pack_type, onnx_type)                                                                           \
  template <typename T, typename... Ts>                                                                                  \
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, pack_type>::value>::type                          \
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) { \
    input_types.push_back(onnx_type);                                                                                    \
    ParseArgs<Ts...>(input_types, output_types);                                                                         \
  }                                                                                                                      \
  template <typename T, typename... Ts>                                                                                  \
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, std::optional<pack_type>>::value>::type           \
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) { \
    input_types.push_back(onnx_type);                                                                                    \
    ParseArgs<Ts...>(input_types, output_types);                                                                         \
  }

#define PARSE_INPUT(data_type, onnx_type)                       \
  PARSE_INPUT_BASE(const Custom::Tensor<data_type>*, onnx_type) \
  PARSE_INPUT_BASE(const Custom::Tensor<data_type>&, onnx_type) \
  PARSE_INPUT_BASE(const Custom::Span<data_type>*, onnx_type)   \
  PARSE_INPUT_BASE(const Custom::Span<data_type>&, onnx_type)   \
  PARSE_INPUT_BASE(data_type, onnx_type)

#define PARSE_OUTPUT(data_type, onnx_type)                                                                                      \
  template <typename T, typename... Ts>                                                                                         \
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom::Tensor<data_type>*>::value>::type                \
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {        \
    output_types.push_back(onnx_type);                                                                                          \
    ParseArgs<Ts...>(input_types, output_types);                                                                                \
  }                                                                                                                             \
  template <typename T, typename... Ts>                                                                                         \
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom::Tensor<data_type>&>::value>::type                \
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {        \
    output_types.push_back(onnx_type);                                                                                          \
    ParseArgs<Ts...>(input_types, output_types);                                                                                \
  }                                                                                                                             \
  template <typename T, typename... Ts>                                                                                         \
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, std::optional<Custom::Tensor<data_type>*>>::value>::type \
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {        \
    output_types.push_back(onnx_type);                                                                                          \
    ParseArgs<Ts...>(input_types, output_types);                                                                                \
  }

#define PARSE_ARGS(data_type, onnx_type) \
  PARSE_INPUT(data_type, onnx_type)      \
  PARSE_OUTPUT(data_type, onnx_type)

  PARSE_ARGS(bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
#if ORT_API_VERSION >= 16
  PARSE_ARGS(MFloat16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
  PARSE_ARGS(BFloat16, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
#endif
  PARSE_ARGS(float, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
  PARSE_ARGS(double, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
  PARSE_ARGS(int8_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
  PARSE_ARGS(int16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)
  PARSE_ARGS(int32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
  PARSE_ARGS(int64_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
  PARSE_ARGS(uint8_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
  PARSE_ARGS(uint16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)
  PARSE_ARGS(uint32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)
  PARSE_ARGS(uint64_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64)
  PARSE_ARGS(std::string, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
  PARSE_ARGS(std::string_view, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)  // todo - remove string_view output

  OrtLiteCustomOp(const char* op_name,
                  const char* execution_provider) : op_name_(op_name),
                                                    execution_provider_(execution_provider) {
    // Zero out OrtCustomOp so that any added func pointers are nullptr for forwards compatibility
    memset(&this->version, 0, sizeof(OrtCustomOp));

    int act_ver = GetActiveOrtAPIVersion();
    OrtCustomOp::version = act_ver < ORT_API_VERSION ? act_ver : ORT_API_VERSION;

    OrtCustomOp::GetName = [](const OrtCustomOp* op) { return static_cast<const OrtLiteCustomOp*>(op)->op_name_.c_str(); };
    OrtCustomOp::GetExecutionProviderType = [](const OrtCustomOp* op) { return ((OrtLiteCustomOp*)op)->execution_provider_.c_str(); };

    OrtCustomOp::GetInputTypeCount = [](const OrtCustomOp* op) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return self->input_types_.size();
    };

    OrtCustomOp::GetInputType = [](const OrtCustomOp* op, size_t indice) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return self->input_types_[indice];
    };

    OrtCustomOp::GetOutputTypeCount = [](const OrtCustomOp* op) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return self->output_types_.size();
    };

    OrtCustomOp::GetOutputType = [](const OrtCustomOp* op, size_t indice) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return self->output_types_[indice];
    };

#if ORT_API_VERSION >= 14
    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp* op, size_t) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return (self->input_types_.empty() || self->input_types_[0] != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_VARIADIC;
    };

    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp* op, size_t) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return (self->output_types_.empty() || self->output_types_[0] != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_VARIADIC;
    };

    OrtCustomOp::GetVariadicInputMinArity = [](const OrtCustomOp*) {
      return 1;
    };

    OrtCustomOp::GetVariadicInputHomogeneity = [](const OrtCustomOp*) {
      return 0;
    };

    OrtCustomOp::GetVariadicOutputMinArity = [](const OrtCustomOp*) {
      return 1;
    };

    OrtCustomOp::GetVariadicOutputHomogeneity = [](const OrtCustomOp*) {
      return 0;
    };

    OrtCustomOp::GetInputMemoryType = [](const OrtCustomOp*, size_t) {
      return OrtMemTypeDefault;
    };
#else
    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp*, size_t) {
      return INPUT_OUTPUT_OPTIONAL;
    };

    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp* op, size_t) {
      return INPUT_OUTPUT_OPTIONAL;
    };
#endif
  }

  const std::string op_name_;
  const std::string execution_provider_;

  std::vector<ONNXTensorElementDataType> input_types_;
  std::vector<ONNXTensorElementDataType> output_types_;
};

template <typename... Args>
struct OrtLiteCustomFunc : public OrtLiteCustomOp {
  using ComputeFn = void (*)(Args...);
  using MyType = OrtLiteCustomFunc<Args...>;

  struct Kernel {
    ComputeFn compute_fn_{};
    std::string ep_{};
    std::unique_ptr<OrtW::CustomOpApi> api_;
  };

  OrtLiteCustomFunc(const char* op_name,
                    const char* execution_provider,
                    ComputeFn compute_fn) : OrtLiteCustomOp(op_name, execution_provider),
                                            compute_fn_(compute_fn) {
    ParseArgs<Args...>(input_types_, output_types_);

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) {
      auto kernel = reinterpret_cast<Kernel*>(op_kernel);
      std::vector<TensorPtr> tensors;
      auto t = CreateTuple<0, 0, Args...>(kernel->api_.get(),
                                          context,
                                          tensors,
                                          kernel->api_->KernelContext_GetInputCount(context),
                                          kernel->api_->KernelContext_GetOutputCount(context),
                                          kernel->ep_);
      std::apply([kernel](Args const&... t_args) { kernel->compute_fn_(t_args...); }, t);
    };

    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi* ort_api, const OrtKernelInfo* info) {
      auto kernel = std::make_unique<Kernel>();
      auto self = static_cast<const OrtLiteCustomFunc*>(this_);
      kernel->compute_fn_ = self->compute_fn_;
      kernel->ep_ = self->execution_provider_;
      kernel->api_ = std::make_unique<OrtW::CustomOpApi>(*ort_api);
      return reinterpret_cast<void*>(kernel.release());
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) {
      delete reinterpret_cast<Kernel*>(op_kernel);
    };
  }

  ComputeFn compute_fn_;
};

class OrtAttributeReader {
 public:
  OrtAttributeReader(const OrtApi& ort_api, const OrtKernelInfo& info) : base_kernel_(ort_api, info) {
  }

  template <class T>
  T TryToGetAttributeWithDefault(const char* name, const T& default_value) const noexcept {
    return base_kernel_.TryToGetAttributeWithDefault(name, default_value);
  }

 private:
  BaseKernel base_kernel_;
};

template <typename CustomOp>
struct OrtLiteCustomStruct : public OrtLiteCustomOp {
  template <typename... Args>
  using CustomComputeFn = void (CustomOp::*)(Args...) const;
  using MyType = OrtLiteCustomStruct<CustomOp>;

  struct Kernel {
    std::unique_ptr<CustomOp> custom_op_;
    std::string ep_{};
    std::unique_ptr<OrtW::CustomOpApi> api_;
  };

  OrtLiteCustomStruct(const char* op_name,
                      const char* execution_provider) : OrtLiteCustomOp(op_name,
                                                                        execution_provider) {
    init(&CustomOp::Compute);
  }

  template <typename... Args>
  void init(CustomComputeFn<Args...>) {
    ParseArgs<Args...>(input_types_, output_types_);

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) {
      auto kernel = reinterpret_cast<Kernel*>(op_kernel);
      std::vector<TensorPtr> tensors;
      auto t = CreateTuple<0, 0, Args...>(kernel->api_.get(),
                                          context,
                                          tensors,
                                          kernel->api_->KernelContext_GetInputCount(context),
                                          kernel->api_->KernelContext_GetOutputCount(context),
                                          kernel->ep_);
      std::apply([kernel](Args const&... t_args) { kernel->custom_op_->Compute(t_args...); }, t);
    };

    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi* ort_api, const OrtKernelInfo* info) {
      auto kernel = std::make_unique<Kernel>();

      if constexpr (std::is_constructible<CustomOp, const OrtApi&, const OrtKernelInfo&>::value) {
        kernel->custom_op_ = std::make_unique<CustomOp>(*ort_api, *info);
      } else {
        kernel->custom_op_ = std::make_unique<CustomOp>(OrtAttributeReader(*ort_api, *info));
      }
      auto self = static_cast<const MyType*>(this_);
      kernel->ep_ = self->execution_provider_;
      kernel->api_ = std::make_unique<OrtW::CustomOpApi>(*ort_api);
      return reinterpret_cast<void*>(kernel.release());
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) {
      delete reinterpret_cast<Kernel*>(op_kernel);
    };
  }
};

template <typename... Args>
OrtLiteCustomOp* CreateLiteCustomOp(const char* op_name,
                                    const char* execution_provider,
                                    void (*custom_compute_fn)(Args...)) {
  using LiteOp = OrtLiteCustomFunc<Args...>;
  return std::make_unique<LiteOp>(op_name, execution_provider, custom_compute_fn).release();
}

template <typename CustomOp>
OrtLiteCustomOp* CreateLiteCustomOp(const char* op_name,
                                    const char* execution_provider) {
  using LiteOp = OrtLiteCustomStruct<CustomOp>;
  return std::make_unique<LiteOp>(op_name, execution_provider).release();
}

}  // namespace Custom
}  // namespace Ort
