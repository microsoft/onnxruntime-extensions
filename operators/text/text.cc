#include "text/op_equal.hpp"
#include "text/op_ragged_tensor.hpp"
#include "text/string_hash.hpp"
#include "text/string_join.hpp"
#include "text/string_lower.hpp"
#include "text/string_split.hpp"
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
      BuildCustomOp(CustomOpStringRegexReplace),
      BuildCustomOp(CustomOpStringRegexSplitWithOffsets),
#endif  // ENABLE_RE2_REGEX
      /*LiteCustomOpStruct("RaggedTensorToSparse", KernelRaggedTensorToSparse),
      LiteCustomOpStruct("RaggedTensorToDense", KernelRaggedTensorToDense),
      LiteCustomOpStruct("StringRaggedTensorToDense", KernelStringRaggedTensorToDense),*/
      LiteCustomOpStruct("StringEqual", KernelStringEqual),
      LiteCustomOp("StringToHashBucket", string_hash),
      LiteCustomOp("StringToHashBucketFast", string_hash_fast),
      LiteCustomOp("StringJoin", string_join),
      LiteCustomOp("StringLower", string_lower),
      LiteCustomOp("StringUpper", string_upper),
      LiteCustomOpStruct("StringMapping", KernelStringMapping),
      LiteCustomOp("MaskedFill", masked_fill),
      LiteCustomOp("StringSplit", string_split),
      LiteCustomOpStruct("StringToVector", KernelStringToVector),
      LiteCustomOpStruct("VectorToString", KernelVectorToString),
      LiteCustomOp("StringLength", string_length),
      LiteCustomOp("StringConcat", string_concat),
      LiteCustomOpStruct("StringECMARegexReplace", KernelStringECMARegexReplace),
      LiteCustomOpStruct("StringECMARegexSplitWithOffsets", KernelStringECMARegexSplitWithOffsets));
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Text = TextLoader;
