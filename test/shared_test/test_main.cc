// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include <filesystem>
#include <fstream>
#include <vector>

#include "gtest/gtest.h"

#define TEST_MAIN main

#if defined(__APPLE__)
  #include <TargetConditionals.h>
  #if TARGET_OS_SIMULATOR || TARGET_OS_IOS
    #undef TEST_MAIN
    #define TEST_MAIN main_no_link_  // there is a UI test app for iOS.
  #endif
#endif

// currently this is the only place with a try/catch. Move the macros to common code if that changes.
#ifdef OCOS_NO_EXCEPTIONS
#define OCOS_TRY if (true)
#define OCOS_CATCH(x) else if (false)
#define OCOS_RETHROW
// In order to ignore the catch statement when a specific exception (not ... ) is caught and referred
// in the body of the catch statements, it is necessary to wrap the body of the catch statement into
// a lambda function. otherwise the exception referred will be undefined and cause build break
#define OCOS_HANDLE_EXCEPTION(func)
#else
#define OCOS_TRY try
#define OCOS_CATCH(x) catch (x)
#define OCOS_RETHROW throw;
#define OCOS_HANDLE_EXCEPTION(func) func()
#endif


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

int TEST_MAIN(int argc, char** argv) {
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
