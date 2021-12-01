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
#endif // ENABLE_RE2_REGEX


FxLoadCustomOpFactory LoadCustomOpClasses_Text = 
    LoadCustomOpClasses<CustomOpClassBegin, 
#if defined(ENABLE_RE2_REGEX)
                        CustomOpStringRegexReplace,
                        CustomOpStringRegexSplitWithOffsets,
#endif // ENABLE_RE2_REGEX
                        CustomOpRaggedTensorToDense,
                        CustomOpRaggedTensorToSparse,
                        CustomOpStringRaggedTensorToDense,
                        CustomOpStringEqual,
                        CustomOpStringHash,
                        CustomOpStringHashFast,
                        CustomOpStringJoin,
                        CustomOpStringLower,
                        CustomOpStringUpper,
                        CustomOpStringMapping,
                        CustomOpMaskedFill,
                        CustomOpStringSplit,
                        CustomOpStringToVector,
                        CustomOpVectorToString,
                        CustomOpStringLength,
                        CustomOpStringConcat,
                        CustomOpStringECMARegexReplace,
                        CustomOpStringECMARegexSplitWithOffsets
                        >;
