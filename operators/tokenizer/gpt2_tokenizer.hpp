#include <list>
#include "ocos.h"
#include "ustring.h"

class VocabData;

struct KernelBpeTokenizer : BaseKernel {
  KernelBpeTokenizer(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context);

 private:
  std::vector<int64_t> Tokenize(const ustring& input, int64_t max_length);

  int64_t padding_length_;
  std::list<int> byte_list_;
  std::shared_ptr<VocabData> bbpe_tokenizer_;
};

struct CustomOpBpeTokenizer : OrtW::CustomOpBase<CustomOpBpeTokenizer, KernelBpeTokenizer> {
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};
