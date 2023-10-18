#include "text/string_functions.h"
#include "text/op_ragged_tensor.hpp"
#include "text/string_to_vector.hpp"
#include "text/vector_to_string.hpp"
#include "text/string_ecmaregex_replace.hpp"
#include "text/string_ecmaregex_split.hpp"
#include "text/string_mapping.hpp"

#if defined(ENABLE_RE2_REGEX)
#include "text/re2_strings/string_regex_replace.hpp"
#include "text/re2_strings/string_regex_split.hpp"
#endif  // ENABLE_RE2_REGEX


FxLoadCustomOpFactory LoadCustomOpClasses_Text = []()-> CustomOpArray& {
  static OrtOpLoader op_loader(
#if defined(ENABLE_RE2_REGEX)
      CustomCpuStruct("StringRegexReplace", KernelStringRegexReplace),
      CustomCpuFunc("StringRegexSplitWithOffsets", KernelStringRegexSplitWithOffsets),
#endif  // ENABLE_RE2_REGEX
      CustomCpuStruct("RaggedTensorToSparse", KernelRaggedTensoroSparse),
      CustomCpuStruct("RaggedTensorToDense", KernelRaggedTensoroDense),
      CustomCpuStruct("StringRaggedTensorToDense", KernelStringRaggedTensoroDense),
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
      CustomCpuStruct("StringMapping", KernelStringMapping),
      CustomCpuStruct("StringToVector", KernelStringToVector),
      CustomCpuStruct("VectorToString", KernelVectorToString),
      CustomCpuStruct("StringECMARegexReplace", KernelStringECMARegexReplace),
      CustomCpuStruct("StringECMARegexSplitWithOffsets", KernelStringECMARegexSplitWithOffsets));

  return op_loader.GetCustomOps();
};
