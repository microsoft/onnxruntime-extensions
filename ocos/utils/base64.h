// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Freely inspired from https://en.wikibooks.org/wiki/Algorithm_Implementation/Miscellaneous/Base64
#pragma once
#include <string>
#include <vector>

bool base64_encode(const std::vector<uint8_t>& input, std::string& encoded);
bool base64_decode(const std::string& encoded, std::vector<uint8_t>& raw);
