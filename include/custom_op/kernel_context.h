#pragma once
#include <optional>
#include <numeric>
#include <type_traits>
#include "onnxruntime_c_api.h"

namespace Ort {
namespace Custom {

// this is for the ORT custom op template magic
struct Arg {
  virtual ~Arg() = default;
};

class KernelContext : public Arg{
public:
  virtual void* AllocScratchBuffer(size_t size) = 0;
  virtual void FreeScratchBuffer(void* p) = 0;
  // TODO: threadpool?
};

#ifdef USE_CUDA
class CUDAKernelContext :  public KernelContext {
public:
  virtual void* AllocCudaScratchBuffer(size_t size) = 0;
  virtual void FreeCudaScratchBuffer(void* p) = 0;
  virtual void* GetCudaStream() const = 0;
  virtual void* GetCublasHandle() const = 0;
  virtual int GetCudaDeviceId() const = 0;
  virtual void* GetScratchBufferUnderMultiStream(const OrtMemoryInfo* , size_t ) { return nullptr; }
};
#endif

// TODO: helper func to create context from global ORT env.

}
}
