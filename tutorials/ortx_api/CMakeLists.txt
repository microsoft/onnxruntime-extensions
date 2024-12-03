cmake_minimum_required(VERSION 3.28)

project(ortx_api_test)

set(CMAKE_CXX_STANDARD 17)
include(FetchContent)

FetchContent_Declare(
  ortx
  GIT_REPOSITORY https://github.com/microsoft/onnxruntime-extensions.git
  GIT_TAG main)

set(OCOS_BUILD_PRESET token_api_only)
FetchContent_MakeAvailable(ortx)

file(GLOB_RECURSE SOURCES "src/*.cc")
add_executable(ortx_api_test ${SOURCES})
target_link_libraries(ortx_api_test onnxruntime_extensions)
target_include_directories(ortx_api_test PRIVATE ${ortx_SOURCE_DIR}/include)
