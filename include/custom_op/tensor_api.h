#pragma once
#include <optional>
#include <numeric>
#include <type_traits>
#include "onnxruntime_f16.h"
#include "ortdevice.h"
#include "kernel_context.h"

namespace Ort {
namespace Custom {

template <typename T>
struct Span {
  const T* data_ = {};
  size_t size_ = {};
  void Assign(const T* data, size_t size) {
    data_ = data;
    size_ = size;
  }
  size_t size() const { return size_; }
  T operator[](size_t indice) const {
    return data_[indice];
  }
  const T* data() const { return data_; }
};


#if ORT_API_VERSION >= 16

template <>
struct Span<MFloat16> {
  const MFloat16* data_ = {};
  size_t size_ = {};
  void Assign(const MFloat16* data, size_t size) {
    data_ = data;
    size_ = size;
  }
  size_t size() const { return size_; }
  MFloat16 operator[](size_t indice) const {
    return data_[indice];
  }
  const MFloat16* data() const { return data_; }
};

template <>
struct Span<BFloat16> {
  const BFloat16* data_ = {};
  size_t size_ = {};
  void Assign(const BFloat16* data, size_t size) {
    data_ = data;
    size_ = size;
  }
  size_t size() const { return size_; }
  BFloat16 operator[](size_t indice) const {
    return data_[indice];
  }
  const BFloat16* data() const { return data_; }
};

#endif

class ITensorStorage{
public:
  virtual const std::vector<int64_t>& Shape() const = 0;
  virtual const void* DataRaw() const = 0;
  virtual void* MutableDataRaw() const = 0;
  virtual bool IsInitialized() const = 0;
  virtual void* Initialize(const std::vector<int64_t>& shape, size_t element_size) = 0;
  virtual OrtDevice Device() const = 0;
  virtual void* Release() = 0;
  virtual ~ITensorStorage() = default;
};


class IAllocator {
public:
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
};


class OrtEagerTensorStorage : public ITensorStorage {
public:
  OrtEagerTensorStorage(const std::vector<int64_t>& shape,
                        void* buffer, OrtDevice device) : buffer_(buffer), shape_(shape), device_(device) {}

  OrtEagerTensorStorage(IAllocator* allocator, OrtDevice device) : allocator_(allocator), device_(device) {}

  virtual ~OrtEagerTensorStorage(){
    if (allocator_ && buffer_)
      allocator_->Free(buffer_);
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
    return buffer_;
  }

  void* MutableDataRaw() const override {
    return buffer_;
  }

  void* Initialize(const std::vector<int64_t>& shape, size_t element_size) override {
    if (IsInitialized())
      return buffer_;
    assert(allocator_);
    shape_ = shape;
    int64_t n_elem = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    auto buffer_size = n_elem * element_size;
    buffer_ = allocator_->Alloc(buffer_size);
    return buffer_;
  }

  OrtDevice Device() const override {
    return device_;
  }

  void* Release() override {
    void* tmp = buffer_;
    buffer_ = 0;
    shape_ = std::nullopt;
    return tmp;
  }

private:
  void* buffer_ {};
  std::optional<std::vector<int64_t>> shape_;
  // caller need to make sure the allocator is alive
  IAllocator* allocator_{};
  OrtDevice device_;
};

template <typename TT>
ONNXTensorElementDataType GetOrtDType(){
  if constexpr (std::is_same<TT, bool>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  else if constexpr (std::is_same<TT, float>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  else if constexpr (std::is_same<TT, double>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  else if constexpr (std::is_same<TT, uint8_t>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  else if constexpr (std::is_same<TT, int8_t>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  else if constexpr (std::is_same<TT, uint16_t>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  else if constexpr (std::is_same<TT, int16_t>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  else if constexpr (std::is_same<TT, uint32_t>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
  else if constexpr (std::is_same<TT, int32_t>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  else if constexpr (std::is_same<TT, uint64_t>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
  else if constexpr (std::is_same<TT, int64_t>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  else if constexpr (std::is_same<TT, std::string>::value)
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  ORTX_CXX_API_THROW("Unexpected type", ORT_RUNTIME_EXCEPTION);
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

class TensorBase : public Arg {
public:
  virtual ~TensorBase() {}

  virtual ONNXTensorElementDataType Type() const = 0; 
  virtual const std::vector<int64_t>& Shape() const = 0;
  virtual int64_t NumberOfElement() const = 0;
  virtual const void* DataRaw() const = 0;
  virtual void* MutableDataRaw() const = 0;
  virtual size_t SizeInBytes() const = 0;
};

template <typename T>
class Tensor : public TensorBase {
 public:
  using TT = typename std::remove_reference<T>::type;
  Tensor(std::unique_ptr<ITensorStorage> tensor_storage) : storage_(std::move(tensor_storage)){
  }

  Tensor(const std::vector<int64_t>& shape, void* buffer, OrtDevice device) : Tensor(std::make_unique<OrtEagerTensorStorage>(shape, buffer, device)) {}

  Tensor(IAllocator* allocator, OrtDevice device) : storage_(std::make_unique<OrtEagerTensorStorage>(allocator, device)) {}

  virtual ~Tensor() = default;

  Tensor(const Tensor& src) = delete;

  Tensor& operator=(Tensor src) = delete;

  Tensor(Tensor&& other) : storage_(std::move(other.storage_)) {
    other.storage_ = nullptr;
    other.span_ = {};
  }

  Tensor& operator=(Tensor&& other)
  {
    storage_ = std::move(other.storage_);
    other.span_ = {};
    return *this;
  }

  operator bool() const {
    return storage_ && storage_->IsInitialized();
  }

  ONNXTensorElementDataType Type() const override {
    return GetOrtDType<T>();
  }

  const std::vector<int64_t>& Shape() const override {
    if (!storage_)
      ORTX_CXX_API_THROW("tensor not initialized.", ORT_RUNTIME_EXCEPTION);
    return storage_->Shape();
  }

  int64_t NumberOfElement() const override {
    if (!storage_)
      ORTX_CXX_API_THROW("tensor not initialized.", ORT_RUNTIME_EXCEPTION);
    auto& shape = storage_->Shape();
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  }

  OrtDevice Device() const {
    return storage_->Device();
  }

  std::string Shape2Str() const {
    if (!storage_)
      ORTX_CXX_API_THROW("tensor not initialized.", ORT_RUNTIME_EXCEPTION);
    if (storage_&& storage_->IsInitialized()) {
      std::string shape_str;
      auto& shape = storage_->Shape();
      for (const auto& dim : shape) {
        shape_str.append(std::to_string(dim));
        shape_str.append(", ");
      }
      return shape_str;
    } else {
      return "empty";
    }
  }

  void* Release() {
    if (!storage_)
      ORTX_CXX_API_THROW("tensor not initialized.", ORT_RUNTIME_EXCEPTION);
    span_ = {};
    return storage_->Release();
  }
  
  const TT* Data() const {
    if (!storage_)
      ORTX_CXX_API_THROW("tensor not initialized.", ORT_RUNTIME_EXCEPTION);
#if ORT_API_VERSION >= 16
    if constexpr (std::is_same<TT, MFloat16>::value || std::is_same<TT, BFloat16>::value)
      return reinterpret_cast<const TT*>(storage_->DataRaw());
    else
#endif
      return static_cast<const TT*>(storage_->DataRaw());
  }

  const void* DataRaw() const override {
    if (!storage_)
      ORTX_CXX_API_THROW("tensor not initialized.", ORT_RUNTIME_EXCEPTION);
    return storage_->DataRaw();
  }

  void* MutableDataRaw() const override {
    return storage_->MutableDataRaw();
  }

  size_t SizeInBytes() const override {
    if (!storage_)
      ORTX_CXX_API_THROW("tensor not initialized.", ORT_RUNTIME_EXCEPTION);
    return NumberOfElement() * sizeof(TT);
  }

  TT* Allocate(const std::vector<int64_t>& shape) {
    if (!storage_)
      ORTX_CXX_API_THROW("tensor not initialized.", ORT_RUNTIME_EXCEPTION);
    // it should be OK to allocate multiple times
    void* buffer = storage_->Initialize(shape, sizeof(TT));
#if ORT_API_VERSION >= 16
    if constexpr (std::is_same<TT, MFloat16>::value || std::is_same<TT, BFloat16>::value)
      return reinterpret_cast<TT*>(buffer);
    else
#endif
      return static_cast<TT*>(buffer);
  }

  const Span<T>& AsSpan() {
    if (!storage_)
      ORTX_CXX_API_THROW("tensor not initialized.", ORT_RUNTIME_EXCEPTION);
#if ORT_API_VERSION >= 16
    if constexpr (std::is_same<TT, MFloat16>::value || std::is_same<TT, BFloat16>::value) {
        ORTX_CXX_API_THROW("AsSpan for MFloat16 / BFloat16 not implemented", ORT_RUNTIME_EXCEPTION);
    }
    else{
#endif
      auto& shape = storage_->Shape();
      if (shape.size() != 1) {
        ORTX_CXX_API_THROW("to get a span, shape must be 1-D, actual shape: " + Shape2Str(), ORT_RUNTIME_EXCEPTION);
      }
      span_.Assign(Data(), shape[0]);
      return span_;
#if ORT_API_VERSION >= 16
    }
#endif 
  }

  const T& AsScalar() {
    if (!storage_)
      ORTX_CXX_API_THROW("tensor not initialized.", ORT_RUNTIME_EXCEPTION);
#if ORT_API_VERSION >= 16
    if constexpr (std::is_same<TT, MFloat16>::value || std::is_same<TT, BFloat16>::value) {
      ORTX_CXX_API_THROW("AsScalar for MFloat16 / BFloat16 not implemented", ORT_RUNTIME_EXCEPTION);
    }
    else{
#endif
      auto& shape = storage_->Shape();
      if ((shape.size() == 1 && shape[0] != 1) || shape.size() > 1) {
        ORTX_CXX_API_THROW("to get a scalar, shape must be {1}, actual shape: " + Shape2Str(), ORT_RUNTIME_EXCEPTION);
      }
      return *Data();
#if ORT_API_VERSION >= 16
    }
#endif
  }

 private:
  std::unique_ptr<ITensorStorage> storage_;
  Span<T> span_;
};

template<typename T>
class IStringTensorStorage{
public:
  using strings = std::vector<T>;
  virtual const std::vector<int64_t>& Shape() const = 0;
  virtual const void* DataRaw() const = 0;
  virtual const strings& Data() const = 0;
  virtual bool IsInitialized() const = 0;
  virtual void SetStringOutput(const strings& ss, const std::vector<int64_t>& dims) = 0;
  virtual void SetStringOutput(const std::vector<const char*>& ss, const std::vector<int64_t>& dims) = 0;
};

template<typename T>
class EagerStringTensorStorage : public IStringTensorStorage<T>{
public:
  using strings = std::vector<T>;
  EagerStringTensorStorage(const strings& ss) : input_strings_(ss), shape_(std::vector<int64_t>{static_cast<int64_t>(ss.size())}){}

  EagerStringTensorStorage() {}

  const std::vector<int64_t>& Shape() const override {
    if (!IsInitialized())
      ORTX_CXX_API_THROW("Tensor not initialized", ORT_RUNTIME_EXCEPTION);
    return *shape_;
  }

  virtual const void* DataRaw() const override {
    if (input_strings_.size() != 1) {
      ORTX_CXX_API_THROW("DataRaw() only applies to string scalar", ORT_RUNTIME_EXCEPTION);
    }
    if constexpr (std::is_same<std::string_view, T>::value)
      return reinterpret_cast<const void*>(input_strings_[0].data());
    else
      return reinterpret_cast<const void*>(input_strings_[0].c_str());
  }

  virtual bool IsInitialized() const override {
    return shape_.has_value();
  }
  
  virtual void SetStringOutput(const strings& ss, const std::vector<int64_t>& dims) override {
    if constexpr (std::is_same<std::string_view, T>::value)
      ORTX_CXX_API_THROW("Set output for string view tensor is not supported", ORT_RUNTIME_EXCEPTION);
    input_strings_.assign(ss.begin(), ss.end());
    shape_ = dims;
  }

  const strings& Data() const override {
    return input_strings_;
  }

  virtual void SetStringOutput(const std::vector<const char*>& ss, const std::vector<int64_t>& dims) override {
    if constexpr (std::is_same<std::string_view, T>::value)
      ORTX_CXX_API_THROW("Set output for string view tensor is not supported", ORT_RUNTIME_EXCEPTION);
    
    for (const char* s : ss){
        input_strings_.push_back(s);
    }
    shape_ = dims;
  }

private:
  std::vector<T> input_strings_;  
  std::optional<std::vector<int64_t>> shape_;
};

template <>
class Tensor<std::string> : public TensorBase {
 public:
  using strings = std::vector<std::string>;

  Tensor(std::unique_ptr<IStringTensorStorage<std::string>> storage) : storage_(std::move(storage)) {}

  Tensor(const strings& ss) : storage_(std::make_unique<EagerStringTensorStorage<std::string>>(ss)) {}

  Tensor() : storage_(std::make_unique<EagerStringTensorStorage<std::string>>()) {}

  ONNXTensorElementDataType Type() const override {
    return GetOrtDType<std::string>();
  }

  const strings& Data() const {
    return storage_->Data();
  }

  const std::vector<int64_t>& Shape() const override {
    return storage_->Shape();
  }

  int64_t NumberOfElement() const override {
    auto& shape = storage_->Shape();
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  }

  std::string Shape2Str() const {
    if (storage_->IsInitialized()) {
      std::string shape_str;
      auto& shape = storage_->Shape();
      for (const auto& dim : shape) {
        shape_str.append(std::to_string(dim));
        shape_str.append(", ");
      }
      return shape_str;
    } else {
      return "empty";
    }
  }

  const void* DataRaw() const override {
    return storage_->DataRaw();
  }

  size_t SizeInBytes() const override {
    auto& ss = storage_->Data();
    if (ss.size() != 1) {
      ORTX_CXX_API_THROW("SizeInBytes() only applies to string scalar", ORT_RUNTIME_EXCEPTION);
    }
    return ss[0].size();
  }

  void SetStringOutput(const strings& ss, const std::vector<int64_t>& dims) {
    storage_->SetStringOutput(ss, dims);
  }
  void SetStringOutput(const std::vector<const char*>& ss, const std::vector<int64_t>& dims) {
    storage_->SetStringOutput(ss, dims);
  }
  const Span<std::string>& AsSpan() {
    ORTX_CXX_API_THROW("span for TensorT of string not implemented", ORT_RUNTIME_EXCEPTION);
  }
  const std::string& AsScalar() {
    auto& ss = storage_->Data();
    if (ss.size() != 1) {
      ORTX_CXX_API_THROW("to get a scalar, shape must be {1}, actual shape: " + Shape2Str(), ORT_RUNTIME_EXCEPTION);
    }
    return ss[0];
  }

 private:
  std::unique_ptr<IStringTensorStorage<std::string>> storage_;
};


template <>
class Tensor<std::string_view> : public TensorBase {
 public:
  using strings = std::vector<std::string_view>;

  Tensor(std::unique_ptr<IStringTensorStorage<std::string_view>> storage) : storage_(std::move(storage)) {}

  Tensor(const strings& ss) : storage_(std::make_unique<EagerStringTensorStorage<std::string_view>>(ss)) {}

  ONNXTensorElementDataType Type() const override {
    return GetOrtDType<std::string_view>();
  }

  const strings& Data() const {
    return storage_->Data();
  }

  const std::vector<int64_t>& Shape() const override {
    return storage_->Shape();
  }

  int64_t NumberOfElement() const override {
    auto& shape = storage_->Shape();
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  }

  std::string Shape2Str() const {
    if (storage_->IsInitialized()) {
      std::string shape_str;
      auto& shape = storage_->Shape();
      for (const auto& dim : shape) {
        shape_str.append(std::to_string(dim));
        shape_str.append(", ");
      }
      return shape_str;
    } else {
      return "empty";
    }
  }

  const void* DataRaw() const override {
    return storage_->DataRaw();
  }

  size_t SizeInBytes() const override {
    auto& ss = storage_->Data();
    if (ss.size() != 1) {
      ORTX_CXX_API_THROW("SizeInBytes() only applies to string scalar", ORT_RUNTIME_EXCEPTION);
    }
    return ss[0].size();
  }

  void SetStringOutput(const strings& ss, const std::vector<int64_t>& dims) {
    storage_->SetStringOutput(ss, dims);
  }
  void SetStringOutput(const std::vector<const char*>& ss, const std::vector<int64_t>& dims) {
    storage_->SetStringOutput(ss, dims);
  }
  const Span<std::string_view>& AsSpan() {
    ORTX_CXX_API_THROW("span for TensorT of string not implemented", ORT_RUNTIME_EXCEPTION);
  }
  const std::string_view& AsScalar() {
    auto& ss = storage_->Data();
    if (ss.size() != 1) {
      ORTX_CXX_API_THROW("to get a scalar, shape must be {1}, actual shape: " + Shape2Str(), ORT_RUNTIME_EXCEPTION);
    }
    return ss[0];
  }

 private:
  std::unique_ptr<IStringTensorStorage<std::string_view>> storage_;
};


template<typename ...Args>
class NamedArgumentDict{
public:
  using ValueTuple = std::tuple<Args...>;

  NamedArgumentDict(const std::vector<const char*>& keys, const std::tuple<Args...>& args) : entries_(args) {
    for (const char* key : keys){
      names_.push_back(key);
    }
  }

  template<typename T>
  T TryToGetAttributeWithDefault(const char* name, const T& default_value) const {
    return TryToGetAttributeWithDefaultInternal<0>(name, default_value);
  }

private:
  template<size_t I, typename T>
  typename std::enable_if<I == sizeof...(Args), T>::type 
  TryToGetAttributeWithDefaultInternal(const char* name, const T& default_value) const {
    return default_value;
  }
 
  template<size_t I, typename T>
  typename std::enable_if<I < sizeof...(Args), T>::type 
  TryToGetAttributeWithDefaultInternal(const char* name, const T& default_value) const {
    if (names_[I] == name){
     if constexpr (std::is_same<std::tuple_element_t<I, ValueTuple>, T>::value)
       return std::get<I>(entries_);  
     else
       throw std::runtime_error("name matched but type is not");
    }
    return TryToGetAttributeWithDefaultInternal<I+1>(name, default_value);
  }

  std::vector<std::string> names_;
  std::tuple<Args...> entries_;

};

}
}
