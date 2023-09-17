#include "text/op_equal.hpp"
#include "text/op_ragged_tensor.hpp"
#include "text/string_hash.hpp"
#include "text/string_join.hpp"
#include "text/string_lower.hpp"
#include "text/string_split.hpp"
#include "text/string_strip.hpp"
#include "text/string_to_vector.hpp"
#include "text/string_upper.hpp"
#include "text/vector_to_string.hpp"
#include "text/string_length.hpp"
#include "text/string_concat.hpp"
#include "text/string_ecmaregex_replace.hpp"
#include "text/string_ecmaregex_split.hpp"
#include "text/string_mapping.hpp"
#include "text/masked_fill.hpp"

#if defined(ENABLE_RE2_REGEX)
#include "text/re2_strings/string_regex_replace.hpp"
#include "text/re2_strings/string_regex_split.hpp"
#endif  // ENABLE_RE2_REGEX

const std::vector<const OrtCustomOp*>& TextLoader() {
  static OrtOpLoader op_loader(
#if defined(ENABLE_RE2_REGEX)
      //CustomCpuStruct("StringRegexReplace", KernelStringRegexReplace),
      //CustomCpuFunc("StringRegexSplitWithOffsets", KernelStringRegexSplitWithOffsets),
#endif  // ENABLE_RE2_REGEX
      //CustomCpuStruct("RaggedTensorToSparse", KernelRaggedTensoroSparse),
      //CustomCpuStruct("RaggedTensorToDense", KernelRaggedTensoroDense),
      //CustomCpuStruct("StringRaggedTensorToDense", KernelStringRaggedTensoroDense),
      //CustomCpuStruct("StringEqual", KernelStringEqual),
      CustomCpuFunc("StringToHashBucket", string_hash),
      CustomCpuFunc("StringToHashBucketFast", string_hash_fast),
      CustomCpuFunc("StringJoin", string_join),
      CustomCpuFunc("StringLower", string_lower),
      CustomCpuFunc("StringUpper", string_upper),
      //CustomCpuStruct("StringMapping", KernelStringMapping),
      CustomCpuFunc("MaskedFill", masked_fill),
      CustomCpuFunc("StringSplit", string_split),
      CustomCpuFunc("StringStrip", string_strip),
      //CustomCpuStruct("StringToVector", KernelStringToVector),
      //CustomCpuStruct("VectorToString", KernelVectorToString),
      CustomCpuFunc("StringLength", string_length),
      CustomCpuFunc("StringConcat", string_concat)//,
      //CustomCpuStruct("StringECMARegexReplace", KernelStringECMARegexReplace),
      //CustomCpuStruct("StringECMARegexSplitWithOffsets", KernelStringECMARegexSplitWithOffsets)
      );
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Text = TextLoader;

