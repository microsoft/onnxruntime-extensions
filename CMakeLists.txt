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
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CheckCXXCompilerFlag)
include(CheckLanguage)

option(CC_OPTIMIZE "Allow compiler optimizations, Set to OFF to disable" ON)
option(ENABLE_PYTHON "Enable Python component building" OFF)
option(ENABLE_TOKENIZER "Enable the tokenizer building" ON)

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

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_FIND_FRAMEWORK NEVER CACHE STRING "...")
if(NOT "${CMAKE_FIND_FRAMEWORK}" STREQUAL "NEVER")
  message(FATAL_ERROR "CMAKE_FIND_FRAMEWORK is not NEVER")
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/externals)
include(FetchContent)
include(googlere2)
include(farmhash)
FetchContent_GetProperties(googlere2)
FetchContent_GetProperties(farmhash)

if (ENABLE_TOKENIZER)
  include(json)
endif()

file(GLOB TARGET_SRC "./ocos/*.cc")
file(GLOB TARGET_HEADERS "./ocos/*.h*")
file(GLOB TARGET_SRC_KERNELS "./ocos/kernels/*.cc" "./ocos/kernels/*.h*")
file(GLOB TARGET_SRC_PYOPS "./ocos/pyfunc/*.cc" "./ocos/pyfunc/*.h*")
file(GLOB TARGET_SRC_HASH "${farmhash_SOURCE_DIR}/src/farmhash.*")

if (ENABLE_TOKENIZER)
  file(GLOB tok_TARGET_SRC "tokenizer/*.cc")
  list(APPEND TARGET_SRC ${tok_TARGET_SRC})
endif()

if(ENABLE_PYTHON)
  list(APPEND TARGET_SRC "./ocos/ortcustomops_pyd.def")
  set(Python3_FIND_REGISTRY NEVER CACHE STRING "...")
  if(NOT "${Python3_FIND_REGISTRY}" STREQUAL "NEVER")
    message(FATAL_ERROR "Python3_FIND_REGISTRY is not NEVER")
  endif()

  find_package(Python3 COMPONENTS Interpreter Development)
  Python3_add_library(ortcustomops SHARED 
    ${TARGET_SRC}
    ${TARGET_SRC_KERNELS}
    ${TARGET_SRC_PYOPS}
    ${TARGET_HEADERS}
    ${TARGET_SRC_HASH})
  target_compile_definitions(ortcustomops PRIVATE PYTHON_OP_SUPPORT)
  set_source_files_properties(ortcustomops_pyd.def PROPERTIES HEADER_FILE_ONLY TRUE)
else()
  list(APPEND TARGET_SRC "./ocos/ortcustomops.def")
  add_library(ortcustomops SHARED
    ${TARGET_SRC}
    ${TARGET_SRC_KERNELS}
    ${TARGET_HEADERS}
    ${TARGET_SRC_HASH})
  set_source_files_properties(ortcustomops.def PROPERTIES HEADER_FILE_ONLY TRUE)
endif()

set(external_libraries re2)
if (ENABLE_TOKENIZER)
  target_compile_definitions(ortcustomops PRIVATE ENABLE_TOKENIZER)
  list(APPEND external_libraries nlohmann_json::nlohmann_json)
endif()

target_include_directories(ortcustomops PUBLIC
  ${PROJECT_SOURCE_DIR}/includes
  ${PROJECT_SOURCE_DIR}/includes/onnxruntime
  ${PROJECT_SOURCE_DIR}/ocos
  ${googlere2_SOURCE_DIR}
  ${farmhash_SOURCE_DIR}/src)

target_link_libraries(ortcustomops PRIVATE ${external_libraries})

target_compile_definitions(ortcustomops PRIVATE
    ONNX_NAMESPACE=onnx
    ONNX_ML
    NOMINMAX
    FARMHASH_NO_BUILTIN_EXPECT)
target_compile_features(ortcustomops PUBLIC cxx_std_11)

if(ENABLE_PYTHON)
  include(pybind11)
  set(NUMPY_NOT_FOUND false)
  exec_program("${Python3_EXECUTABLE}"
      ARGS "-c \"import numpy; print(numpy.get_include())\""
      OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
      RETURN_VALUE NUMPY_NOT_FOUND)
  if(${NUMPY_NOT_FOUND})
      message(FATAL_ERROR
              "Cannot get NumPy include directory: Is NumPy installed?")
  endif(${NUMPY_NOT_FOUND})

  target_include_directories(ortcustomops PRIVATE
      ${NUMPY_INCLUDE_DIR}
      ${pybind11_INCLUDE_DIRS}
  )
  if(NOT "${OCOS_EXTENTION_NAME}" STREQUAL "")
    set_target_properties(ortcustomops PROPERTIES
      LIBRARY_OUTPUT_NAME ${OCOS_EXTENTION_NAME}
      PREFIX ""
      SUFFIX "")
  endif()
endif()

# test section
include(googletest)
file(GLOB TEST_SRC "${PROJECT_SOURCE_DIR}/test/test*.cc")
add_executable(ortcustomops_test ${TEST_SRC})
target_link_libraries(ortcustomops_test gtest_main ortcustomops ${external_libraries})

add_test(NAME ortcustomops_test COMMAND test)
