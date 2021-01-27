// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "base64.h"
#include <stdexcept>

const static std::string encodeLookup("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/");
const static char padCharacter = '=';

void base64_encode(const std::vector<uint8_t>& input, std::string& encoded) {
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
      temp = (*cursor++) << 16;  //Convert to big endian
      encoded.append(1, encodeLookup[(temp & 0x00FC0000) >> 18]);
      encoded.append(1, encodeLookup[(temp & 0x0003F000) >> 12]);
      encoded.append(2, padCharacter);
      break;
    case 2:
      temp = (*cursor++) << 16;  //Convert to big endian
      temp += (*cursor++) << 8;
      encoded.append(1, encodeLookup[(temp & 0x00FC0000) >> 18]);
      encoded.append(1, encodeLookup[(temp & 0x0003F000) >> 12]);
      encoded.append(1, encodeLookup[(temp & 0x00000FC0) >> 6]);
      encoded.append(1, padCharacter);
      break;
  }
  encoded = encoded;
}

void base64_decode(const std::string& input, std::vector<uint8_t>& decoded) {
  if (input.length() % 4)  //Sanity check
    throw std::runtime_error("Non-Valid base64!");
  size_t padding = 0;
  if (input.length()) {
    if (input[input.length() - 1] == padCharacter)
      padding++;
    if (input[input.length() - 2] == padCharacter)
      padding++;
  }
  //Setup a vector to hold the result
  decoded.clear();
  decoded.reserve(((input.length() / 4) * 3) - padding);
  uint32_t temp = 0;  //Holds decoded quanta
  std::string::const_iterator cursor = input.begin();
  while (cursor < input.end()) {
    for (size_t quantumPosition = 0; quantumPosition < 4; quantumPosition++) {
      temp <<= 6;
      if (*cursor >= 0x41 && *cursor <= 0x5A)  // This area will need tweaking if
        temp |= *cursor - 0x41;                // you are using an alternate alphabet
      else if (*cursor >= 0x61 && *cursor <= 0x7A)
        temp |= *cursor - 0x47;
      else if (*cursor >= 0x30 && *cursor <= 0x39)
        temp |= *cursor + 0x04;
      else if (*cursor == 0x2B)
        temp |= 0x3E;  //change to 0x2D for URL alphabet
      else if (*cursor == 0x2F)
        temp |= 0x3F;                    //change to 0x5F for URL alphabet
      else if (*cursor == padCharacter)  //pad
      {
        switch (input.end() - cursor) {
          case 1:  //One pad character
            decoded.push_back((temp >> 16) & 0x000000FF);
            decoded.push_back((temp >> 8) & 0x000000FF);
            return;
          case 2:  //Two pad characters
            decoded.push_back((temp >> 10) & 0x000000FF);
            return;
          default:
            throw std::runtime_error("Invalid Padding in Base 64!");
        }
      } else
        throw std::runtime_error("Non-Valid Character in Base 64!");
      cursor++;
    }
    decoded.push_back((temp >> 16) & 0x000000FF);
    decoded.push_back((temp >> 8) & 0x000000FF);
    decoded.push_back((temp)&0x000000FF);
  }
}
