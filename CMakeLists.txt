cmake_minimum_required(VERSION 3.16.0)
project(ortcustomops VERSION 0.1.0 LANGUAGES C CXX)

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
include(CheckCXXCompilerFlag)
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 14)

option(CC_OPTIMIZE "Allow compiler optimizations, Set to OFF to disable" ON)
option(ENABLE_PYTHON "Enable Python component building" ON)

if(NOT CC_OPTIMIZE)
  message("!!!THE COMPILER OPTIMIZATION HAS BEEN DISABLED, DEBUG-ONLY!!!")
  string(REGEX REPLACE "([\-\/]O[011123456789])" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "([\-\/]O[011123456789])" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

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

if(ENABLE_PYTHON)
  set(Python3_FIND_REGISTRY NEVER CACHE STRING "...")
  if(NOT "${Python3_FIND_REGISTRY}" STREQUAL "NEVER")
    message(FATAL_ERROR "Python3_FIND_REGISTRY is not NEVER")
  endif()

  find_package(Python3 COMPONENTS Interpreter Development)
endif()

file(GLOB TARGET_SRC "./ocos/*.cc"  "./ocos/ortcustomops.def")
file(GLOB TARGET_SRC_KERNELS "./ocos/kernels/*.cc")
file(GLOB TARGET_SRC_PYOPS "./ocos/pyfunc/*.cc")

if (ENABLE_PYTHON)
  Python3_add_library(ortcustomops SHARED ${TARGET_SRC} ${TARGET_SRC_KERNELS} ${TARGET_SRC_PYOPS})
  target_compile_definitions(ortcustomops PRIVATE PYTHON_OP_SUPPORT)
else()
  add_library(ortcustomops SHARED ${TARGET_SRC} ${TARGET_SRC_KERNELS} ${TARGET_SRC_PYOPS})
endif()

target_include_directories(ortcustomops PUBLIC
  ./includes
  ./includes/onnxruntime)

set_source_files_properties(ortcustomops.def PROPERTIES HEADER_FILE_ONLY TRUE)
target_compile_definitions(ortcustomops PRIVATE ONNX_NAMESPACE=onnx ONNX_ML NOMINMAX)
target_compile_features(ortcustomops PUBLIC cxx_std_11)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/externals)

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

  add_custom_command(
    TARGET ortcustomops POST_BUILD
    COMMAND powershell  -NoProfile -ExecutionPolicy RemoteSigned -Command  ./pysetup.ps1 "$<TARGET_FILE_DIR:ortcustomops>"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

include(googletest)
add_executable(cuop_test ${PROJECT_SOURCE_DIR}/test/test_main.cc)

target_link_libraries(cuop_test gtest_main ortcustomops)

add_test(NAME cuop_test COMMAND test)
