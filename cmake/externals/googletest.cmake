FetchContent_Declare(
  googletest
  URL             https://github.com/google/googletest/archive/refs/tags/v1.15.0.zip
  URL_HASH        SHA1=9d2d0af8d77ac726ea55d44a8fa727ec98311349
  EXCLUDE_FROM_ALL
)

set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
set_target_properties(gtest PROPERTIES FOLDER "externals/google")
