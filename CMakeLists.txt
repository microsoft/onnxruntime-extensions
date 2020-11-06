cmake_minimum_required(VERSION 3.16.0)
project(ortcustomops VERSION 0.1.0 LANGUAGES C CXX)
# set(CMAKE_VERBOSE_MAKEFILE ON)

# Enable CTest
enable_testing()
include(CTest)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo." FORCE)
endif()

set(ONNX_ML 1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(REPO_ROOT ${PROJECT_SOURCE_DIR})
include(CheckCXXCompilerFlag)
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 14)

option(CC_OPTIMIZE "Allow compiler optimizations, Set to OFF to disable" ON)
option(ENABLE_PYTHON "Enable Python component building" ON)

if(NOT CC_OPTIMIZE)
  message("!!!THE COMPILER OPTIMIZATION HAS BEEN DISABLED, DEBUG-ONLY!!!")
  string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
  string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

  if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Od")
  endif()
endif()

set(CMAKE_FIND_FRAMEWORK NEVER CACHE STRING "...")
if(NOT "${CMAKE_FIND_FRAMEWORK}" STREQUAL "NEVER")
  message(FATAL_ERROR "CMAKE_FIND_FRAMEWORK is not NEVER")
endif()

# external dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/externals)

# Build ortcustomops
include(googlere2)
if (ENABLE_PYTHON)
  # External dependencies
  include(${PROJECT_SOURCE_DIR}/cmake/ortcustom_python.cmake)
endif()

# static build
file(GLOB TARGET_SRC "./ocos/*.cc" "./ocos/*.h*")
file(GLOB TARGET_SRC_KERNELS "./ocos/kernels/*.cc" "./ocos/kernels/*.h*")
add_library(ortcustomops_lib STATIC ${TARGET_SRC} ${TARGET_SRC_KERNELS})
target_include_directories(ortcustomops_lib PUBLIC 
  ./includes
  ./includes/onnxruntime
  ${googlere2_SOURCE_DIR})
target_compile_definitions(ortcustomops_lib PRIVATE ONNX_NAMESPACE=onnx ONNX_ML NOMINMAX)
target_compile_features(ortcustomops_lib PUBLIC cxx_std_11)

#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)  

# Unit tests
include(googletest)
FetchContent_GetProperties(googlere2)

file(GLOB TEST_SRC "${PROJECT_SOURCE_DIR}/test/test*.cc")
add_executable(ortcustomops_test ${TEST_SRC})
target_include_directories(ortcustomops_test PRIVATE 
  ./includes
  ./includes/onnxruntime
  ${googlere2_SOURCE_DIR})
target_link_libraries(ortcustomops_test
  gtest_main
  ${googlere2_BINARY_DIR}/${CMAKE_BUILD_TYPE}/re2.lib
  ortcustomops_lib)

add_test(NAME ortcustomops_test COMMAND test)
