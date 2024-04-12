FetchContent_Declare(
  googletest
  URL         https://github.com/google/googletest/archive/530d5c8c84abd2a46f38583ee817743c9b3a42b4.zip
  URL_HASH    SHA1=5e3a61db2aa975cfd0f97ba92c818744e7fa7034
)

FetchContent_MakeAvailable(googletest)
set_target_properties(gmock PROPERTIES FOLDER "externals/gtest")
set_target_properties(gmock_main PROPERTIES FOLDER "externals/gtest")
set_target_properties(gtest PROPERTIES FOLDER "externals/gtest")
set_target_properties(gtest_main PROPERTIES FOLDER "externals/gtest")
