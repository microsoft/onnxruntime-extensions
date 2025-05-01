#include <iostream>
#include <string>
#include <vector>

#include "ortx_tokenizer.h"

extError_t tokenize_text(const OrtxTokenizer *tokenizer,
                         const char *text, std::string &decoded_text, std::vector<extTokenId_t> &ids)
{
  OrtxTokenId2DArray *tok_2d_output = NULL;
  const char *tok_input[] = {text};
  extError_t err = OrtxTokenize(tokenizer, tok_input, 1, &tok_2d_output, true);
  if (err != kOrtxOK)
  {
    return err;
  }

  size_t length = 0;
  const extTokenId_t *token_ids = NULL;
  OrtxTokenId2DArrayGetItem(tok_2d_output, 0, &token_ids, &length);

  OrtxStringArray *detok_output = NULL;
  err = OrtxDetokenize1D(tokenizer, token_ids, length, &detok_output);
  if (err != kOrtxOK)
  {
    ORTX_DISPOSE(tok_2d_output);
    return err;
  }
  ids.insert(ids.end(), token_ids, token_ids + length);

  const char *decoded_str = NULL;
  OrtxStringArrayGetItem(detok_output, 0, &decoded_str);
  decoded_text = decoded_str;

  ORTX_DISPOSE(tok_2d_output);
  ORTX_DISPOSE(detok_output);
  return kOrtxOK;
}

int main()
{
  int ver = OrtxGetAPIVersion();
  std::cout << "Ortx API version: " << ver << std::endl;
  OrtxTokenizer *tokenizer = NULL;

  std::cout << "Please specify the tokenizer model file path (like <root>/test/data/llama2)" << std::endl;
  std::string model_path;
  std::cin >> model_path;

  extError_t err = OrtxCreateTokenizer(&tokenizer, model_path.c_str());
  if (err != kOrtxOK)
  {
    std::cerr << "Failed to create tokenizer" << std::endl;
    return 1;
  }

  const char *input = "How many hours does it take a man to eat a Helicopter?";
  std::string decoded_text;
  std::vector<extTokenId_t> ids;
  err = tokenize_text(tokenizer, input, decoded_text, ids);
  if (err != kOrtxOK)
  {
    std::cerr << "Failed to tokenize text" << std::endl;
    return 1;
  }

  std::cout << "Input  : " << input << std::endl;
  // output the token ids
  std::cout << "Token IDs: ";
  for (const auto &id : ids)
  {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  std::cout << "Decoded: " << decoded_text << std::endl;

  OrtxDisposeOnly(tokenizer); // Clean up the tokenizer
  return 0;
}
