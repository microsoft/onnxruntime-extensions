#include <string>
#include <unordered_map>
#include <vector>
#include <string>

#include "kernels/kernels.h"


struct KernelBpeTokenizer : BaseKernel {
  KernelBpeTokenizer(OrtApi api)
  :BaseKernel(apt){
  }

  void Compute(OrtKernelContext* context){
      ;}
      
  static void SetupKernel() {
      // Load the tokenizer JSON file.
  }
  
};


struct CustomOpBpeTokenizer : Ort::CustomOpBase<CustomOpBpeTokenizer, KernelBpeTokenizer> {
  CustomOpBpeTokenizer() {
        KernelBpeTokenizer::SetupKernel();
  }
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) {
    return new KernelBpeTokenizer(api);
  }

  const char* GetName() const {
    return "BPETokenizer";
  }

  size_t GetInputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  }
  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
};


const OrtCustomOp** LoadTextDomainSchemaList()
{
    static CustomOpBpeTokenizer c_CoBpeTokenizer;
    static const OrtCustomOp* c_DomainList[] = {&c_CoBpeTokenizer, nullptr};
    return c_DomainList;
}
