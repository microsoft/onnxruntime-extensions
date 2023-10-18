#include "text/string_functions.h"
#include "text/op_ragged_tensor.hpp"
#include "text/string_to_vector.hpp"
#include "text/vector_to_string.hpp"
#include "text/string_ecmaregex_replace.hpp"
#include "text/string_ecmaregex_split.hpp"
#include "text/string_mapping.hpp"

#if defined(ENABLE_RE2_REGEX)
#include "text/re2_strings/string_regex.h"
#endif  // ENABLE_RE2_REGEX


FxLoadCustomOpFactory LoadCustomOpClasses_Text = []()-> CustomOpArray& {
  static OrtOpLoader op_loader(
#if defined(ENABLE_RE2_REGEX)
      CustomCpuStructV2("StringRegexReplace", KernelStringRegexReplace),
      CustomCpuFuncV2("StringRegexSplitWithOffsets", KernelStringRegexSplitWithOffsets),
#endif  // ENABLE_RE2_REGEX
      CustomCpuFuncV2("RaggedTensorToSparse", RaggedTensorToSparse),
      CustomCpuStructV2("RaggedTensorToDense", KernelRaggedTensoroDense),
      CustomCpuFuncV2("StringRaggedTensorToDense", StringRaggedTensorToDense),
      CustomCpuFuncV2("StringEqual", string_equal),
      CustomCpuFuncV2("StringToHashBucket", string_hash),
      CustomCpuFuncV2("StringToHashBucketFast", string_hash_fast),
      CustomCpuFuncV2("StringJoin", string_join),
      CustomCpuFuncV2("StringLower", string_lower),
      CustomCpuFuncV2("StringUpper", string_upper),
      CustomCpuFuncV2("MaskedFill", masked_fill),
      CustomCpuFuncV2("StringSplit", string_split),
      CustomCpuFuncV2("StringStrip", string_strip),
      CustomCpuFuncV2("StringLength", string_length),
      CustomCpuFuncV2("StringConcat", string_concat),
      CustomCpuStructV2("StringMapping", KernelStringMapping),
      CustomCpuStructV2("StringToVector", KernelStringToVector),
      CustomCpuStructV2("VectorToString", KernelVectorToString),
      CustomCpuStructV2("StringECMARegexReplace", KernelStringECMARegexReplace),
      CustomCpuStructV2("StringECMARegexSplitWithOffsets", KernelStringECMARegexSplitWithOffsets));

  return op_loader.GetCustomOps();
};
