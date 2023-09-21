// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.hpp"
#include <fstream>

namespace ort_extensions {
namespace test {
std::vector<uint8_t> LoadBytesFromFile(const std::filesystem::path& filename) {
  using namespace std;
  ifstream ifs(filename, ios::binary | ios::ate);
  ifstream::pos_type pos = ifs.tellg();

  std::vector<uint8_t> input_bytes(pos);
  ifs.seekg(0, ios::beg);
  // we want uint8_t values so reinterpret_cast so we don't have to read chars and copy to uint8_t after.
  ifs.read(reinterpret_cast<char*>(input_bytes.data()), pos);

  return input_bytes;
}
}  // namespace test
}  // namespace ort_extensions
