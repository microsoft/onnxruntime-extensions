#include <kernels/ustring.hpp>
#include "kernels/kernels.h"
#include "utils/string_utils.h"

class VocabData;

struct KernelBpeTokenizer : BaseKernel {
  KernelBpeTokenizer(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 private:
  std::vector<int64_t> Tokenize(const ustring& input, int64_t max_length);

  int64_t padding_length_;
  std::list<int> byte_list_;
  std::shared_ptr<VocabData> bbpe_tokenizer_;
};

struct CustomOpBpeTokenizer : Ort::CustomOpBase<CustomOpBpeTokenizer, KernelBpeTokenizer> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};