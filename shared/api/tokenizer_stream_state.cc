#include "tokenizer_stream_state.h"

namespace ort_extensions {

DetokenizerCache::DetokenizerCache() : OrtxObjectImpl(kOrtxKindDetokenizerCache) {}
DetokenizerCache::~DetokenizerCache() = default;

}  // namespace ort_extensions