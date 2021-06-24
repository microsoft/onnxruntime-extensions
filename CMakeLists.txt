cmake_minimum_required(VERSION 3.16.0)
project(onnxruntime_extensions LANGUAGES C CXX)
# set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo." FORCE)
endif()


set(CPACK_PACKAGE_NAME "onnxruntime_extensions")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "3")
set(CPACK_PACKAGE_VERSION_PATCH "1")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CheckCXXCompilerFlag)
include(CheckLanguage)

option(CC_OPTIMIZE "Allow compiler optimizations, Set to OFF to disable" ON)
option(OCOS_ENABLE_PYTHON "Enable Python component building" OFF)
option(OCOS_ENABLE_CTEST "Enable C++ test" OFF)
option(OCOS_ENABLE_TF_STRING "Enable String Operator Set" ON)
option(OCOS_ENABLE_GPT2_TOKENIZER "Enable the GPT2 tokenizer building" ON)
option(OCOS_ENABLE_SPM_TOKENIZER "Enable the SentencePiece tokenizer building" ON)
option(OCOS_ENABLE_BERT_TOKENIZER "Enable the BertTokenizer building" ON)
option(OCOS_ENABLE_BLINGFIRE "Enable the Blingfire building" ON)
option(OCOS_ENABLE_MATH "Enable the math tensor operators building" ON)
option(OCOS_ENABLE_STATIC_LIB "Enable generating static library" OFF)


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
  if (NOT TARGET re2::re2)
    set(RE2_BUILD_TESTING OFF CACHE INTERNAL "")
    message(STATUS "Fetch googlere2")
    include(googlere2)
    FetchContent_GetProperties(googlere2)
  endif()

  if (NOT TARGET farmhash)
    message(STATUS "Fetch farmhash")
    include(farmhash)
    FetchContent_GetProperties(farmhash)
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set_property(TARGET re2 PROPERTY COMPILE_OPTIONS )
  endif()
endif()

file(GLOB TARGET_SRC "operators/*.cc" "operators/*.h")
if (OCOS_ENABLE_TF_STRING)
  file(GLOB TARGET_SRC_KERNELS "operators/text/*.cc" "operators/text/*.h*")
  file(GLOB TARGET_SRC_HASH "${farmhash_SOURCE_DIR}/src/farmhash.*")
  list(APPEND TARGET_SRC ${TARGET_SRC_KERNELS} ${TARGET_SRC_HASH})
endif()

if (OCOS_ENABLE_MATH)
  set(DLIB_NO_GUI_SUPPORT ON CACHE INTERNAL "")
  set(DLIB_USE_CUDA OFF CACHE INTERNAL "")
  set(DLIB_USE_LAPACK OFF CACHE INTERNAL "")
  set(DLIB_USE_BLAS OFF CACHE INTERNAL "")
  include(dlib)
  file(GLOB TARGET_SRC_MATH "operators/math/*.cc" "operators/math/*.h*")
  list(APPEND TARGET_SRC ${TARGET_SRC_MATH})
endif()

if (OCOS_ENABLE_GPT2_TOKENIZER)
  # GPT2
  if (NOT TARGET nlohmann_json)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    message(STATUS "Fetch json")
    include(json)
  endif()

  file(GLOB tok_TARGET_SRC "operators/tokenizer/gpt*.cc" "operators/tokenizer/unicode*.*")
  list(APPEND TARGET_SRC ${tok_TARGET_SRC})
endif()

if (OCOS_ENABLE_SPM_TOKENIZER)
  # SentencePiece
  set(SPM_ENABLE_TCMALLOC OFF CACHE INTERNAL "")
  set(SPM_ENABLE_SHARED OFF CACHE INTERNAL "")
  message(STATUS "Fetch sentencepiece")
  include(sentencepieceproject)
  file(GLOB stpiece_TARGET_SRC "operators/tokenizer/sentencepiece/*.cc" "operators/tokenizer/sentencepiece*")
  list(REMOVE_ITEM stpiece_TARGET_SRC INCLUDE REGEX ".*((spm)|(train)).*")
  list(APPEND TARGET_SRC ${stpiece_TARGET_SRC})
endif()

if (OCOS_ENABLE_BERT_TOKENIZER)
  # Bert
  file(GLOB bert_TARGET_SRC "operators/tokenizer/wordpiece*.*")
  list(APPEND TARGET_SRC ${bert_TARGET_SRC})
endif()

if (OCOS_ENABLE_BLINGFIRE)
  file(GLOB blingfire_TARGET_SRC "operators/tokenizer/blingfire*.*")
  list(APPEND TARGET_SRC ${blingfire_TARGET_SRC})
endif()


add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
add_library(ocos_operators STATIC ${TARGET_SRC})
target_include_directories(ocos_operators PUBLIC operators/tokenizer)

set(ocos_libraries ocos_operators)
if (OCOS_ENABLE_TF_STRING)
  list(APPEND ocos_libraries re2)
endif()

target_include_directories(ocos_operators PUBLIC
  ${PROJECT_SOURCE_DIR}/includes
  ${PROJECT_SOURCE_DIR}/includes/onnxruntime
  ${PROJECT_SOURCE_DIR}/operators)

set(OCOS_COMPILE_DEFINITIONS "")

if (OCOS_ENABLE_TF_STRING)
  target_include_directories(ocos_operators PUBLIC
    ${googlere2_SOURCE_DIR}
    ${farmhash_SOURCE_DIR}/src)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_TF_STRING)
endif()

if (OCOS_ENABLE_MATH)
  target_include_directories(ocos_operators PUBLIC ${dlib_SOURCE_DIR})
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_MATH)
  # The dlib matrix implementation is all in the headers, no library compiling needed.
  if (WIN32)
    list(APPEND ocos_libraries dlib::dlib)
  endif()
endif()

if (OCOS_ENABLE_GPT2_TOKENIZER)
  # GPT2
  target_include_directories(ocos_operators PRIVATE ${json_SOURCE_DIR}/single_include)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_GPT2_TOKENIZER)
  list(APPEND ocos_libraries nlohmann_json::nlohmann_json)
endif()

if (OCOS_ENABLE_SPM_TOKENIZER)
  # SentencePiece
  target_include_directories(ocos_operators PUBLIC ${sentencepieceproject_INCLUDE_DIRS})
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_SPM_TOKENIZER)
  list(APPEND ocos_libraries sentencepiece-static)
endif()

if (OCOS_ENABLE_BLINGFIRE)
  include(blingfire)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_BLINGFIRE)
  list(APPEND ocos_libraries bingfirtinydll_static)
endif()

if (OCOS_ENABLE_TF_STRING)
  target_compile_definitions(ocos_operators PRIVATE
      NOMINMAX
      FARMHASH_NO_BUILTIN_EXPECT)
endif()

if (OCOS_ENABLE_BERT_TOKENIZER)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_BERT_TOKENIZER)
endif()

target_compile_definitions(ocos_operators PRIVATE ${OCOS_COMPILE_DEFINITIONS})

file(GLOB shared_TARGET_SRC "shared/*.cc" "shared/*.h")
if(OCOS_ENABLE_PYTHON)
  set(Python3_FIND_REGISTRY NEVER CACHE STRING "...")
  if(NOT "${Python3_FIND_REGISTRY}" STREQUAL "NEVER")
    message(FATAL_ERROR "Python3_FIND_REGISTRY is not NEVER")
  endif()
  find_package(Python3 COMPONENTS Interpreter Development.Module NumPy)
  if (NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 or NumPy not found!")
  endif()
  if (WIN32)
    list(APPEND shared_TARGET_SRC "${PROJECT_SOURCE_DIR}/onnxruntime_extensions/ortcustomops.def")
  endif()

  file(GLOB TARGET_SRC_PYOPS "pyop/*.cc" "pyop/*.h")
  add_library(ortcustomops SHARED ${TARGET_SRC_PYOPS} ${shared_TARGET_SRC})
  list(APPEND OCOS_COMPILE_DEFINITIONS PYTHON_OP_SUPPORT)
  # building static lib has higher priority
elseif(OCOS_ENABLE_STATIC_LIB)
  add_library(ortcustomops STATIC ${shared_TARGET_SRC})
elseif(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  add_executable(ortcustomops ${shared_TARGET_SRC})
  set_target_properties(ortcustomops PROPERTIES LINK_FLAGS "                                  \
                      -s WASM=1                                                               \
                      -s NO_EXIT_RUNTIME=0                                                    \
                      -s ALLOW_MEMORY_GROWTH=1                                                \
                      -s SAFE_HEAP=0                                                          \
                      -s MODULARIZE=1                                                         \
                      -s SAFE_HEAP_LOG=0                                                      \
                      -s STACK_OVERFLOW_CHECK=0                                               \
                      -s EXPORT_ALL=0                                                         \
                      -s VERBOSE=0                                                            \
                      --no-entry")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_property(TARGET ortcustomops APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=1 -s DEMANGLE_SUPPORT=1")
  else()
    set_property(TARGET ortcustomops APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=0 -s DEMANGLE_SUPPORT=0")
  endif()
else()
  list(APPEND shared_TARGET_SRC "${PROJECT_SOURCE_DIR}/shared/ortcustomops.def")
  add_library(ortcustomops SHARED ${shared_TARGET_SRC})
endif()

target_compile_definitions(ortcustomops PRIVATE  ${OCOS_COMPILE_DEFINITIONS})
target_link_libraries(ortcustomops PRIVATE ${ocos_libraries})

if(OCOS_ENABLE_PYTHON)
  message(STATUS "Fetch pybind11")
  include(pybind11)
  target_include_directories(ortcustomops PRIVATE
    $<TARGET_PROPERTY:Python3::Module,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:Python3::NumPy,INTERFACE_INCLUDE_DIRECTORIES>
    ${pybind11_INCLUDE_DIRS}
  )

  target_compile_definitions(ortcustomops PRIVATE
    $<TARGET_PROPERTY:Python3::Module,INTERFACE_COMPILE_DEFINITIONS>)

  target_link_libraries(ortcustomops PRIVATE Python3::Module)

  if(NOT "${OCOS_EXTENTION_NAME}" STREQUAL "")
    if(NOT WIN32)
      set_target_properties(ortcustomops PROPERTIES
        LIBRARY_OUTPUT_NAME ${OCOS_EXTENTION_NAME}
        PREFIX ""
        SUFFIX "")
    endif()
  endif()
endif()


# test section
if (OCOS_ENABLE_CTEST)
  # Enable CTest
  find_library(ONNXRUNTIME onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}")
  if (NOT ONNXRUNTIME)
    message(FATAL_ERROR "The ctest needs the prebuilt onnxruntime libraries directory, please specify it by ONNXRUNTIME_LIB_DIR.")
  endif()

  enable_testing()
  message(STATUS "Fetch CTest")
  include(CTest)

  set(TEST_SRC_DIR ${PROJECT_SOURCE_DIR}/test)
  message(STATUS "Fetch googletest")
  include(googletest)
  file(GLOB static_TEST_SRC "${TEST_SRC_DIR}/static_test/*.cc")
  add_executable(operators_test ${static_TEST_SRC})
  target_link_libraries(operators_test gtest_main ${ocos_libraries})
  add_test(NAME operators_test COMMAND $<TARGET_FILE:operators_test>)

  # needs to link with stdc++fs in Linux
  if(UNIX AND NOT APPLE)
    set(FS_STDLIB stdc++fs)
  endif()

  file(GLOB shared_TEST_SRC "${TEST_SRC_DIR}/shared_test/*.cc")
  add_executable(ortcustomops_test ${shared_TEST_SRC})
  if (ONNXRUNTIME_LIB_DIR)
    target_link_directories(ortcustomops_test PRIVATE ${ONNXRUNTIME_LIB_DIR})
  endif()
  target_link_libraries(ortcustomops_test ortcustomops onnxruntime gtest_main ${ocos_libraries} ${FS_STDLIB})
  if (WIN32)
    file(TO_CMAKE_PATH "${ONNXRUNTIME_LIB_DIR}/*" ONNXRUNTIME_LIB_FILEPATTERN)
    file(GLOB ONNXRUNTIME_LIB_FILES CONFIGURE_DEPENDS "${ONNXRUNTIME_LIB_FILEPATTERN}")
    add_custom_command(
      TARGET ortcustomops_test POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_LIB_FILES} $<TARGET_FILE_DIR:ortcustomops_test>)
  endif()

  set(TEST_DATA_SRC ${TEST_SRC_DIR}/data)
  set(TEST_DATA_DES ${onnxruntime_extensions_BINARY_DIR}/data)

  # Copy test data from source to destination.
  add_custom_command(
    TARGET ortcustomops_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${TEST_DATA_SRC}
    ${TEST_DATA_DES})
  add_test(NAME ortcustomops_test COMMAND $<TARGET_FILE:ortcustomops_test>)
endif()
