// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <vector>

namespace ort_extensions {
namespace test {
std::vector<uint8_t> LoadBytesFromFile(const std::filesystem::path& filename);

}  // namespace test
}  // namespace ort_extensions
