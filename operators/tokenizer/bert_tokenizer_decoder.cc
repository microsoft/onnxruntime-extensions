#include "bert_tokenizer_decoder.hpp"

KernelBertTokenizerDecoder::KernelBertTokenizerDecoder(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api, info) {
  std::string vocab = ort_.KernelInfoGetAttribute<std::string>(info, "vocab_file");
  bool do_lower_case = TryToGetAttributeWithDefault("do_lower_case", true);
  bool do_basic_tokenize = TryToGetAttributeWithDefault("do_basic_tokenize", true);
  std::string unk_token = TryToGetAttributeWithDefault("unk_token", std::string("[UNK]"));
  std::string sep_token = TryToGetAttributeWithDefault("sep_token", std::string("[SEP]"));
  std::string pad_token = TryToGetAttributeWithDefault("pad_token", std::string("[PAD]"));
  std::string cls_token = TryToGetAttributeWithDefault("cls_token", std::string("[CLS]"));
  std::string mask_token = TryToGetAttributeWithDefault("mask_token", std::string("[MASK]"));
  bool tokenize_chinese_chars = TryToGetAttributeWithDefault("tokenize_chinese_chars", true);
  bool strip_accents = TryToGetAttributeWithDefault("strip_accents", false);
  std::string suffix_indicator = TryToGetAttributeWithDefault("suffix_indicator", std::string("##"));

  decoder_ = std::make_shared<BertTokenizerDecoder>(vocab, do_lower_case, do_basic_tokenize, ustring(unk_token),
                                               ustring(sep_token), ustring(pad_token),ustring(cls_token),
                                               ustring(mask_token), tokenize_chinese_chars, strip_accents, ustring(suffix_indicator));
}

void KernelBertTokenizerDecoder::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> input_data;
  GetTensorMutableDataString(api_, ort_, context, input, input_data);

  if (input_data.size() != 1 && input_data.size() != 2) {
    ORT_CXX_API_THROW("[BertTokenizerDecoder]: only support one or two query.", ORT_INVALID_GRAPH);
  }
  std::vector<int64_t> input_ids;
  std::vector<int64_t> token_type_ids;



  std::vector<int64_t> attention_mask(input_ids.size(), 1);

  std::vector<int64_t> output_dim({static_cast<int64_t>(input_ids.size())});

  SetOutput(context, 0, output_dim, input_ids);
  SetOutput(context, 1, output_dim, token_type_ids);
  SetOutput(context, 2, output_dim, attention_mask);
}

void* CustomOpBertTokenizerDecoder::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelBertTokenizerDecoder(api, info);
};

const char* CustomOpBertTokenizerDecoder::GetName() const { return "BertTokenizerDecoder"; };

size_t CustomOpBertTokenizerDecoder::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpBertTokenizerDecoder::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpBertTokenizerDecoder::GetOutputTypeCount() const {
  return 3;
};

ONNXTensorElementDataType CustomOpBertTokenizerDecoder::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};
