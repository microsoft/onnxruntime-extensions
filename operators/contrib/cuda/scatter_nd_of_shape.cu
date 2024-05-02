// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scatter_nd_of_shape.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace contrib {

template <class NTYPE>
NTYPE flattened_dimension(const std::vector<NTYPE>& values, int64_t first = 0) {
  NTYPE r = 1;
  auto end = values.begin() + first;
  for (auto it = values.begin(); it != end; ++it)
    r *= *it;
  return r;
}

#define _ENFORCE(cond, msg) \
  if (!(cond)) ORTX_CXX_API_THROW(msg, ORT_RUNTIME_EXCEPTION);

#ifndef HIP_LONG
#define HIP_LONG int32_t
#endif

#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,  // max threads per block
    maxElementsPerThread = 4,  // max element processed per thread
  };
};

template <typename T>
__device__ __forceinline__ void _add_inplace(T& x, const T a) { x += a; }

template <>
__device__ __forceinline__ void _add_inplace(half& x, const half a) {
#if __CUDA_ARCH__ < 700
  x = __float2half(__half2float(x) + __half2float(a));
#else
  x += a;
#endif
}

template <typename T>
__global__ void
addition_inplace_kernel(T* __restrict__ output_data, const int64_t* __restrict__ indices_data,
                        const T* __restrict__ updates_data, const CUDA_LONG indice_size,
                        const CUDA_LONG nrows, const CUDA_LONG stride) {
  HIP_LONG id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= stride)
    return;

  for (size_t i = 0; i < nrows; ++i) {
    output_data[i * stride + id] = 0;
  }

  for (size_t i = 0; i < indice_size; ++i) {
    _add_inplace(output_data[indices_data[i] * stride + id], updates_data[i * stride + id]);
  }
}

//////////////////
// ScatterNDOfShapeOp...
//////////////////

template <typename T>
void* ScatterNDOfShapeOp<T>::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  return std::make_unique<ScatterNDOfShapeKernel<T>>(api, info).release();
}

template <typename T>
const char* ScatterNDOfShapeOp<T>::GetName() const {
  return "ScatterNDOfShape";
}

template <typename T>
const char* ScatterNDOfShapeOp<T>::GetExecutionProviderType() const {
  return "CUDAExecutionProvider";
}

template <typename T>
size_t ScatterNDOfShapeOp<T>::GetInputTypeCount() const { return 3; };

template <>
ONNXTensorElementDataType ScatterNDOfShapeOp<float>::GetInputType(std::size_t index) const {
  switch (index) {
    case 0:
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    default:
      ORTX_CXX_API_THROW("Wrong input index.", ORT_RUNTIME_EXCEPTION);
  }
}

template <>
ONNXTensorElementDataType ScatterNDOfShapeOp<ortc::MFloat16>::GetInputType(std::size_t index) const {
  switch (index) {
    case 0:
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    default:
      ORTX_CXX_API_THROW("Wrong input index.", ORT_RUNTIME_EXCEPTION);
  }
}

template <typename T>
OrtMemType ScatterNDOfShapeOp<T>::GetInputMemoryType(std::size_t index) const {
  switch (index) {
    case 0:
      return OrtMemTypeCPUInput;
    case 1:
    case 2:
      return OrtMemTypeDefault;
    default:
      ORTX_CXX_API_THROW("Wrong input index.", ORT_RUNTIME_EXCEPTION);
  }
}

template <typename T>
OrtCustomOpInputOutputCharacteristic
ScatterNDOfShapeOp<T>::GetInputCharacteristic(std::size_t index) const {
  switch (index) {
    case 0:
    case 1:
    case 2:
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    default:
      ORTX_CXX_API_THROW("Wrong output index.", ORT_RUNTIME_EXCEPTION);
  }
}

template <typename T>
size_t ScatterNDOfShapeOp<T>::GetOutputTypeCount() const { return 1; }

template <>
ONNXTensorElementDataType ScatterNDOfShapeOp<float>::GetOutputType(std::size_t index) const {
  // D, scale D
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    default:
      ORTX_CXX_API_THROW("Wrong output index.", ORT_RUNTIME_EXCEPTION);
  }
}

template <>
ONNXTensorElementDataType ScatterNDOfShapeOp<ortc::MFloat16>::GetOutputType(std::size_t index) const {
  // D, scale D
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    default:
      ORTX_CXX_API_THROW("Wrong output index.", ORT_RUNTIME_EXCEPTION);
  }
}

template <typename T>
OrtCustomOpInputOutputCharacteristic
ScatterNDOfShapeOp<T>::GetOutputCharacteristic(std::size_t index) const {
  switch (index) {
    case 0:
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    default:
      ORTX_CXX_API_THROW("Wrong output index", ORT_RUNTIME_EXCEPTION);
  }
}

///////////////////
// ScatterNDOfShapeKernel
///////////////////

template <typename T>
ScatterNDOfShapeKernel<T>::ScatterNDOfShapeKernel(const OrtApi& api,
                                                  const OrtKernelInfo* info) {
  char value_string[1000];
  std::size_t size = 1000;
  Ort::ThrowOnError(api.KernelInfoGetAttribute_string(info, "reduction", value_string, &size));
  std::string value = value_string;
  if (value == "add")
    reduction_ = Reduction::Add;
  else
    ORTX_CXX_API_THROW("unexpected reduction", ORT_RUNTIME_EXCEPTION);

  cudaDeviceProp prop;
  int deviceId = 0;
  cudaGetDeviceProperties(&prop, deviceId);
  maxThreadPerBlock_ = prop.maxThreadsPerBlock;
}

template <typename T>
void ScatterNDOfShapeKernel<T>::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  int n_inputs = ctx.GetInputCount();
  _ENFORCE(n_inputs == 3, "Expecting 3 inputs.");
  Ort::ConstValue shape = ctx.GetInput(0);
  Ort::ConstValue indices = ctx.GetInput(1);
  Ort::ConstValue updates = ctx.GetInput(2);
  Ort::UnownedValue output;

  std::vector<int64_t> dimensions = shape.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> indices_shape = indices.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> updates_shape = updates.GetTensorTypeAndShapeInfo().GetShape();
  _ENFORCE(dimensions.size() == 1, "Shape must be a 1-dimension tensor.");

  cudaStream_t stream = (cudaStream_t)ctx.GetGPUComputeStream();

  auto memi = updates.GetTensorMemoryInfo();
  _ENFORCE(memi.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU, "Tensor updates is not on GPU.");
  auto mem = shape.GetTensorMemoryInfo();
  _ENFORCE(mem.GetDeviceType() == OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU, "Input shape is not on CPU.");
  const int64_t* X = shape.GetTensorData<int64_t>();
  std::vector<int64_t> dims(X, X + dimensions[0]);
  output = ctx.GetOutput(0, dims);

  std::vector<int64_t> input_shape = output.GetTensorTypeAndShapeInfo().GetShape();

  if (reduction_ == Reduction::Add &&
      indices_shape[indices_shape.size() - 1] == 1 && input_shape.size() == 2 &&
      input_shape[input_shape.size() - 1] >= maxThreadPerBlock_) {
    size_t indice_size = static_cast<size_t>(flattened_dimension(indices_shape));
    size_t update_size = static_cast<size_t>(flattened_dimension(updates_shape));
    _ENFORCE(update_size == indice_size * input_shape[input_shape.size() - 1], "Size mismatch.");
    ComputeNoAtomic(stream, input_shape, indices_shape, output.GetTensorMutableData<T>(), indices.GetTensorData<int64_t>(), updates.GetTensorData<T>());
  } else {
    ORTX_CXX_API_THROW("This operator can only be used when the indices_shape[-1] == 1 and input_shape is a 2D matrix.", ORT_RUNTIME_EXCEPTION);
  }
}

template <typename T>
void _ComputeNoAtomic(cudaStream_t stream, const std::vector<int64_t>& input_shape,
                      const std::vector<int64_t>& indices_shape, T* output_data,
                      const int64_t* indices_data, const T* updates_data,
                      int threads_per_block, int blocks_per_grid, size_t indice_size, size_t nrows, size_t stride) {
  dim3 threads(threads_per_block);
  dim3 blocks(blocks_per_grid);
  addition_inplace_kernel<T><<<blocks, threads, 0, stream>>>(output_data, indices_data, updates_data, indice_size, nrows, stride);
}

template <>
void _ComputeNoAtomic<ortc::MFloat16>(cudaStream_t stream, const std::vector<int64_t>& input_shape,
                                      const std::vector<int64_t>& indices_shape, ortc::MFloat16* output_data,
                                      const int64_t* indices_data, const ortc::MFloat16* updates_data,
                                      int threads_per_block, int blocks_per_grid, size_t indice_size, size_t nrows, size_t stride) {

  dim3 threads(threads_per_block);
  dim3 blocks(blocks_per_grid);
  addition_inplace_kernel<half><<<blocks, threads, 0, stream>>>((half*)output_data, indices_data, (const half*)updates_data, indice_size, nrows, stride);
}

template <typename T>
void ScatterNDOfShapeKernel<T>::ComputeNoAtomic(cudaStream_t& stream,
                                                const std::vector<int64_t>& input_shape,
                                                const std::vector<int64_t>& indices_shape,
                                                T* output_data, const int64_t* indices_data,
                                                const T* updates_data) const {
  // The kernel is slow if there are a lot of duplicates.
  // reduction_ == Reduction::add
  // indices_shape[indices_shape.size() - 1] == 1
  // input_shape.size() == 2
  size_t indice_size = static_cast<size_t>(flattened_dimension(indices_shape));
  size_t input_size = static_cast<size_t>(flattened_dimension(input_shape));
  size_t stride = input_shape[input_shape.size() - 1];
  size_t nrows = input_size / stride;

  std::vector<size_t> next_batch(indice_size);
  std::vector<uint8_t> processed(input_shape[0], 0);
  std::vector<uint8_t> processed_once(input_shape[0], 0);

  int threads_per_block = std::min(256, maxThreadPerBlock_ / 8);
  int blocks_per_grid = (stride + threads_per_block - 1) / threads_per_block;
  _ComputeNoAtomic(stream, input_shape, indices_shape, output_data, indices_data, updates_data, threads_per_block, blocks_per_grid, indice_size, nrows, stride);
}

static ScatterNDOfShapeOp<float> _op32;
static ScatterNDOfShapeOp<ortc::MFloat16> _op16;

}  // namespace contrib
