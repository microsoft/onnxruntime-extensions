// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <vector>

#include "gtest/gtest.h"

#include "exceptions.h"

namespace {
void FixCurrentDir() {
  // adjust for the Google Test Adapter in Visual Studio not setting the current path to $(ProjectDir),
  // which results in us being 2 levels below where the `data` folder is copied to and where the extensions
  // library is
  auto cur = std::filesystem::current_path();

  do {
    auto data_dir = cur / "data";

    if (std::filesystem::exists(data_dir) && std::filesystem::is_directory(data_dir)) {
      break;
    }

    cur = cur.parent_path();
    ASSERT_NE(cur, cur.root_path()) << "Reached root directory without finding 'data' directory.";
  } while (true);

  // set current path as the extensions library is also loaded from that directory by TestInference
  std::filesystem::current_path(cur);
}
}  // namespace

int main(int argc, char** argv) {
  int status = 0;

  OCOS_TRY {
    ::testing::InitGoogleTest(&argc, argv);

    FixCurrentDir();
    status = RUN_ALL_TESTS();
  }
  OCOS_CATCH(const std::exception& ex) {
    OCOS_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      status = -1;
    });
  }

  return status;
}
