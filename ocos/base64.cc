// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "base64.h"
#include <stdexcept>

const static std::string encodeLookup("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/");
const static char padCharacter = '=';

bool base64_encode(const std::vector<uint8_t>& input, std::string& encoded) {
  encoded.clear();
  encoded.reserve(((input.size() / 3) + (input.size() % 3 > 0)) * 4);
  uint32_t temp;
  std::vector<uint8_t>::const_iterator cursor = input.begin();
  for (size_t idx = 0; idx < input.size() / 3; idx++) {
    temp = (*cursor++) << 16;  //Convert to big endian
    temp += (*cursor++) << 8;
    temp += (*cursor++);
    encoded.append(1, encodeLookup[(temp & 0x00FC0000) >> 18]);
    encoded.append(1, encodeLookup[(temp & 0x0003F000) >> 12]);
    encoded.append(1, encodeLookup[(temp & 0x00000FC0) >> 6]);
    encoded.append(1, encodeLookup[(temp & 0x0000003F)]);
  }
  switch (input.size() % 3) {
    case 1:
      temp = (*cursor++) << 16;
      encoded.append(1, encodeLookup[(temp & 0x00FC0000) >> 18]);
      encoded.append(1, encodeLookup[(temp & 0x0003F000) >> 12]);
      encoded.append(2, padCharacter);
      break;
    case 2:
      temp = (*cursor++) << 16;
      temp += (*cursor++) << 8;
      encoded.append(1, encodeLookup[(temp & 0x00FC0000) >> 18]);
      encoded.append(1, encodeLookup[(temp & 0x0003F000) >> 12]);
      encoded.append(1, encodeLookup[(temp & 0x00000FC0) >> 6]);
      encoded.append(1, padCharacter);
      break;
  }
  encoded = encoded;
  return true;
}

bool base64_decode(const std::string& input, std::vector<uint8_t>& decoded) {
  if (input.length() % 4)
    return false;
  size_t padding = 0;
  if (input.length()) {
    if (input[input.length() - 1] == padCharacter)
      padding++;
    if (input[input.length() - 2] == padCharacter)
      padding++;
  }

  decoded.clear();
  decoded.reserve(((input.length() / 4) * 3) - padding);
  uint32_t temp = 0;
  std::string::const_iterator cursor = input.begin();
  size_t quantumPosition;
  while (cursor != input.end()) {
    for (quantumPosition = 0; quantumPosition < 4; ++quantumPosition) {
      temp <<= 6;
      if (*cursor >= 0x41 && *cursor <= 0x5A)
        temp |= *cursor - 0x41;
      else if (*cursor >= 0x61 && *cursor <= 0x7A)
        temp |= *cursor - 0x47;
      else if (*cursor >= 0x30 && *cursor <= 0x39)
        temp |= *cursor + 0x04;
      else if (*cursor == 0x2B)
        temp |= 0x3E;
      else if (*cursor == 0x2F)
        temp |= 0x3F;
      else if (*cursor == padCharacter) {
        switch (input.end() - cursor) {
          case 1:  //One pad character
            decoded.push_back((temp >> 16) & 0x000000FF);
            decoded.push_back((temp >> 8) & 0x000000FF);
            return true;
          case 2:  //Two pad characters
            decoded.push_back((temp >> 10) & 0x000000FF);
            return true;
          default:
            return false;
        }
      } else
        return false;
      ++cursor;
    }
    decoded.push_back((temp >> 16) & 0x000000FF);
    decoded.push_back((temp >> 8) & 0x000000FF);
    decoded.push_back((temp)&0x000000FF);
  }
  return true;
}
