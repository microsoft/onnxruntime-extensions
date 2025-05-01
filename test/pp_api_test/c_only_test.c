// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* C-only file, to verify the header file C compatibility */

#include <stdio.h>
#include <string.h>
#include "c_only_test.h"

#if defined(_WIN32)
#define strdup _strdup
#endif


extError_t tokenize_text(OrtxTokenizer* tokenizer, const char* text, char** decoded_text) {
  OrtxTokenId2DArray* tok_2d_output = NULL;
  const char* tok_input[] = {text};
  extError_t err = OrtxTokenize(tokenizer, tok_input, 1, &tok_2d_output, true);
  if (err != kOrtxOK) {
    return err;
  }

  size_t length = 0;
  const extTokenId_t* token_ids = NULL;
  OrtxTokenId2DArrayGetItem(tok_2d_output, 0, &token_ids, &length);

  OrtxStringArray* detok_output = NULL;
  err = OrtxDetokenize1D(tokenizer, token_ids, length, &detok_output);
  if (err != kOrtxOK) {
    ORTX_DISPOSE(tok_2d_output);
    return err;
  }

  const char* decoded_str = NULL;
  OrtxStringArrayGetItem(detok_output, 0, &decoded_str);
  *decoded_text = strdup(decoded_str);

  ORTX_DISPOSE(tok_2d_output);
  ORTX_DISPOSE(detok_output);
  return kOrtxOK;
}
