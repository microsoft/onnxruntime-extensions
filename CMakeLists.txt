cmake_minimum_required(VERSION 3.16.0)
project(ortcustomops VERSION 0.1.0 LANGUAGES C CXX)
# set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CheckCXXCompilerFlag)
include(CheckLanguage)

option(CC_OPTIMIZE "Allow compiler optimizations, Set to OFF to disable" ON)
option(OCOS_ENABLE_PYTHON "Enable Python component building" OFF)
option(OCOS_ENABLE_TF_STRING "Enable String Operator Set" ON)
option(OCOS_ENABLE_GPT2_TOKENIZER "Enable the GPT2 tokenizer building" ON)
option(OCOS_ENABLE_SPM_TOKENIZER "Enable the SentencePiece tokenizer building" ON)
option(OCOS_ENABLE_BERT_TOKENIZER "Enable the BertTokenizer building" ON)

find_library(ONNXRUNTIME onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}")
if ((NOT OCOS_ENABLE_PYTHON) AND (NOT ONNXRUNTIME))
  message(FATAL_ERROR "Cannot find onnxruntime in the default library paths, please specify the ONNXRUNTIME_LIB_DIR.")
endif()

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

# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/externals)
include(FetchContent)
if (OCOS_ENABLE_TF_STRING)
  set(RE2_BUILD_TESTING OFF CACHE INTERNAL "")
  message(STATUS "fetch googlere2")
  include(googlere2)
  message(STATUS "fetch farmhash")
  include(farmhash)
  FetchContent_GetProperties(googlere2)
  FetchContent_GetProperties(farmhash)
endif()

file(GLOB TARGET_SRC "./ocos/*.cc" "./ocos/*.h*"  "./ocos/utils/*.h*" "./ocos/utils/*.cc")
if (OCOS_ENABLE_TF_STRING)
  file(GLOB TARGET_SRC_KERNELS "./ocos/kernels/*.cc" "./ocos/kernels/*.h*")
  file(GLOB TARGET_SRC_HASH "${farmhash_SOURCE_DIR}/src/farmhash.*")
  list(APPEND TARGET_SRC ${TARGET_SRC_KERNELS} ${TARGET_SRC_HASH})
endif()

if (OCOS_ENABLE_GPT2_TOKENIZER)
  # GPT2
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  message(STATUS "fetch json")
  include(json)
  file(GLOB tok_TARGET_SRC "tokenizer/gpt*.cc" "tokenizer/unicode*.*")
  list(APPEND TARGET_SRC ${tok_TARGET_SRC})
endif()

if (OCOS_ENABLE_SPM_TOKENIZER)
  # SentencePiece
  set(SPM_ENABLE_TCMALLOC OFF CACHE INTERNAL "")
  set(SPM_ENABLE_SHARED OFF CACHE INTERNAL "")
  message(STATUS "fetch sentencepiece")
  include(sentencepieceproject)
  file(GLOB stpiece_TARGET_SRC "sentencepiece/*.cc" "tokenizer/sentencepiece*")
  list(REMOVE_ITEM stpiece_TARGET_SRC INCLUDE REGEX ".*((spm)|(train)).*")
  list(APPEND TARGET_SRC ${stpiece_TARGET_SRC})
endif()

if (OCOS_ENABLE_BERT_TOKENIZER)
  # Bert
  file(GLOB bert_TARGET_SRC "tokenizer/wordpiece*.*")
  list(APPEND TARGET_SRC ${bert_TARGET_SRC})
endif()

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
add_library(ortcustomops_static STATIC ${TARGET_SRC})

set(ocos_libraries ortcustomops_static)
if (OCOS_ENABLE_TF_STRING)
  list(APPEND ocos_libraries re2)
endif()

target_include_directories(ortcustomops_static PUBLIC
  ${PROJECT_SOURCE_DIR}/includes
  ${PROJECT_SOURCE_DIR}/includes/onnxruntime
  ${PROJECT_SOURCE_DIR}/ocos)

set(OCOS_COMPILE_DEFINITIONS "")

if (OCOS_ENABLE_TF_STRING)
  target_include_directories(ortcustomops_static PUBLIC
    ${googlere2_SOURCE_DIR}
    ${farmhash_SOURCE_DIR}/src)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_TF_STRING)
endif()

if (OCOS_ENABLE_GPT2_TOKENIZER)
  # GPT2
  target_include_directories(ortcustomops_static PRIVATE ${json_SOURCE_DIR}/single_include)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_GPT2_TOKENIZER)
  list(APPEND ocos_libraries nlohmann_json::nlohmann_json)
endif()

if (OCOS_ENABLE_SPM_TOKENIZER)
  # SentencePiece
  target_include_directories(ortcustomops_static PRIVATE ${PROJECT_SOURCE_DIR}/tokenizer ${sentencepieceproject_INCLUDE_DIRS})
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_SPM_TOKENIZER)
  list(APPEND ocos_libraries sentencepiece-static)
endif()

if (OCOS_ENABLE_TF_STRING)
  target_compile_definitions(ortcustomops_static PRIVATE
      NOMINMAX
      FARMHASH_NO_BUILTIN_EXPECT)
endif()

if (OCOS_ENABLE_BERT_TOKENIZER)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_BERT_TOKENIZER)
endif()

target_compile_definitions(ortcustomops_static PRIVATE ${OCOS_COMPILE_DEFINITIONS})

file(GLOB shared_TARGET_SRC "shared/*.cc" "shared/*.h")
if(OCOS_ENABLE_PYTHON)
  file(GLOB TARGET_SRC_PYOPS "./ocos/pyfunc/*.cc" "./ocos/pyfunc/*.h*")
  set(Python3_FIND_REGISTRY NEVER CACHE STRING "...")
  if(NOT "${Python3_FIND_REGISTRY}" STREQUAL "NEVER")
    message(FATAL_ERROR "Python3_FIND_REGISTRY is not NEVER")
  endif()
  message("-- Find Python3")
  find_package(Python3 COMPONENTS Interpreter Development)
  message("-- Found Python3")

  if (WIN32)
    list(APPEND shared_TARGET_SRC "${PROJECT_SOURCE_DIR}/onnxruntime_customops/ortcustomops.def")
  endif()

  Python3_add_library(ortcustomops SHARED ${TARGET_SRC_PYOPS} ${shared_TARGET_SRC})
  list(APPEND OCOS_COMPILE_DEFINITIONS PYTHON_OP_SUPPORT)
else()
  list(APPEND shared_TARGET_SRC "${PROJECT_SOURCE_DIR}/shared/ortcustomops.def")
  add_library(ortcustomops SHARED ${shared_TARGET_SRC})
endif()

target_compile_definitions(ortcustomops PRIVATE  ${OCOS_COMPILE_DEFINITIONS})
if (OCOS_ENABLE_SPM_TOKENIZER)   # FIXME: this include path is not recommendeded.
  target_include_directories(ortcustomops PRIVATE ${PROJECT_SOURCE_DIR}/tokenizer ${sentencepieceproject_INCLUDE_DIRS})
endif()
target_link_libraries(ortcustomops PRIVATE ${ocos_libraries})

if(OCOS_ENABLE_PYTHON)
  message(STATUS "fetch pybind11")
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
    if(NOT WIN32)
      set_target_properties(ortcustomops PROPERTIES
        LIBRARY_OUTPUT_NAME ${OCOS_EXTENTION_NAME}
        PREFIX ""
        SUFFIX "")
    endif()
  endif()
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
message(STATUS "fetch CPack")
include(CPack)

# test section
if (NOT OCOS_ENABLE_PYTHON)
  # Enable CTest
  enable_testing()
  message(STATUS "fetch CTest")
  include(CTest)

  set(TEST_SRC_DIR ${PROJECT_SOURCE_DIR}/test)
  message(STATUS "fetch googletest")
  include(googletest)
  file(GLOB static_TEST_SRC "${TEST_SRC_DIR}/static_test/*.cc")
  add_executable(ortcustomops_static_test ${static_TEST_SRC})
  target_link_libraries(ortcustomops_static_test gtest_main ${ocos_libraries})
  add_test(NAME ortcustomops_static_test COMMAND $<TARGET_FILE:ortcustomops_static_test>)

  file(GLOB shared_TEST_SRC "${TEST_SRC_DIR}/shared_test/*.cc")
  add_executable(ortcustomops_test ${shared_TEST_SRC})
  if (ONNXRUNTIME_LIB_DIR)
    target_link_directories(ortcustomops_test PRIVATE ${ONNXRUNTIME_LIB_DIR})
    target_link_libraries(ortcustomops_test ortcustomops onnxruntime gtest_main ${ocos_libraries})
    if (WIN32)
      file(TO_CMAKE_PATH "${ONNXRUNTIME_LIB_DIR}/*" ONNXRUNTIME_LIB_FILEPATTERN)
      file(GLOB ONNXRUNTIME_LIB_FILES CONFIGURE_DEPENDS "${ONNXRUNTIME_LIB_FILEPATTERN}")
      add_custom_command(
        TARGET ortcustomops_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_LIB_FILES} $<TARGET_FILE_DIR:ortcustomops_test>)
    endif()
  endif()

  set(TEST_DATA_SRC ${TEST_SRC_DIR}/data)
  set(TEST_DATA_DES ${ortcustomops_BINARY_DIR}/data)

  # Copy test data from source to destination.
  add_custom_command(
    TARGET ortcustomops_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${TEST_DATA_SRC}
    ${TEST_DATA_DES})
  add_test(NAME ortcustomops_test COMMAND $<TARGET_FILE:ortcustomops_test>)
endif()
