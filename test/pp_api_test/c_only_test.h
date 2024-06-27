// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ortx_tokenizer.h"
// make sure the C only compiler compatibility only.
#include "ortx_processor.h"
#include "ortx_extractor.h"


#ifdef __cplusplus
extern "C"
#endif  // __cplusplus
extError_t tokenize_text(OrtxTokenizer* tokenizer, const char* text, char** decoded_text);
