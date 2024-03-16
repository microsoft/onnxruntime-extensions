#include <optional>
#include <numeric>
#include <type_traits>
#include "onnxruntime_customop.hpp"
#include "onnxruntime_f16.h"

namespace Ort {
namespace Custom {

// this is for the ORT custom op template magic
class Arg {
};

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
  virtual bool IsInitialized() const = 0;
  virtual void* Initialize(const std::vector<int64_t>& shape, size_t element_size) = 0;
};

class IAllocator {
public:
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* p) = 0;
};

// TODO: remove this

class TestAllocator : public IAllocator {
public:
  void* Alloc(size_t size) override {
    return malloc(size);
  }

  void Free(void* p) override {
    if (p){
      free(p);
    }
  }
};

class OrtEagerTensorStorage : public ITensorStorage {
public:
  OrtEagerTensorStorage(const std::vector<int64_t>& shape,
                        void* buffer) : buffer_(buffer), shape_(shape){

  }

  OrtEagerTensorStorage(IAllocator* allocator) : allocator_(allocator){
  }

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

private:
  void* buffer_ {};
  std::optional<std::vector<int64_t>> shape_;
  // caller need to make sure the allocator is alive
  IAllocator* allocator_;
};

template <typename T>
class Tensor : public Arg {
 public:
  using TT = typename std::remove_reference<T>::type;
  Tensor(std::unique_ptr<ITensorStorage> tensor_storage) : storage_(std::move(tensor_storage)){
  }

  Tensor(const std::vector<int64_t>& shape, void* buffer) : Tensor(std::make_unique<OrtEagerTensorStorage>(shape, buffer)) {}

  Tensor(IAllocator* allocator) : storage_(std::make_unique<OrtEagerTensorStorage>(allocator)){}

  virtual ~Tensor() = default;

  operator bool() const {
    return storage_->IsInitialized();
  }

  const std::vector<int64_t>& Shape() const {
    return storage_->Shape();
  }

  int64_t NumberOfElement() const {
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
  
  const TT* Data() const {
#if ORT_API_VERSION >= 16
    if constexpr (std::is_same<TT, MFloat16>::value || std::is_same<TT, BFloat16>::value)
      return reinterpret_cast<const TT*>(storage_->DataRaw());
    else
#endif
      return static_cast<const TT*>(storage_->DataRaw());
  }

  const void* DataRaw() const {
    return storage_->DataRaw();
  }

  size_t SizeInBytes() const {
    return NumberOfElement() * sizeof(TT);
  }

  TT* Allocate(const std::vector<int64_t>& shape) {
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

}
}
