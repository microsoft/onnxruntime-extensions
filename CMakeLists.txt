# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.28)
project(onnxruntime_extensions LANGUAGES C CXX)

# set(CMAKE_VERBOSE_MAKEFILE ON)
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo." FORCE)
endif()

function(read_version_file major_var minor_var patch_var)
  set(version_file ${PROJECT_SOURCE_DIR}/version.txt)
  file(READ ${version_file} version_file_content)
  if(version_file_content MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)[\n]?$")
    set(${major_var} ${CMAKE_MATCH_1} PARENT_SCOPE)
    set(${minor_var} ${CMAKE_MATCH_2} PARENT_SCOPE)
    set(${patch_var} ${CMAKE_MATCH_3} PARENT_SCOPE)
  else()
    message(FATAL_ERROR "Failed to parse version from file: ${version_file}")
  endif()
endfunction()

set(CPACK_PACKAGE_NAME "onnxruntime_extensions")
# Configure CPack to generate a NuGet package
set(CPACK_GENERATOR "NuGet")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "onnxruntime-extensions: a pre-/post-processing library for ONNX Runtime")
read_version_file(CPACK_PACKAGE_VERSION_MAJOR CPACK_PACKAGE_VERSION_MINOR CPACK_PACKAGE_VERSION_PATCH)
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
  cmake_policy(SET CMP0077 NEW)
endif()

# Avoid warning of Calling FetchContent_Populate(GSL) is deprecated temporarily
# TODO: find a better way to handle the header-only 3rd party deps
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
  cmake_policy(SET CMP0169 NEW)
endif()

# Needed for Java
set(CMAKE_C_STANDARD 99)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(CMakeDependentOption)

set(_ORTX_STANDALONE_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(_ORTX_STANDALONE_PROJECT ON)
endif()

option(CC_OPTIMIZE "Allow compiler optimizations, Set to OFF to disable" ON)
option(OCOS_ENABLE_PYTHON "Enable Python component building, (deprecated)" OFF)
option(OCOS_ENABLE_CTEST "Enable C++ test" ${_ORTX_STANDALONE_PROJECT})
option(OCOS_ENABLE_CPP_EXCEPTIONS "Enable C++ Exception" ON)
option(OCOS_ENABLE_TF_STRING "Enable String Operator Set" ON)
option(OCOS_ENABLE_RE2_REGEX "Enable StringRegexReplace and StringRegexSplit" ON)
option(OCOS_ENABLE_GPT2_TOKENIZER "Enable the GPT2 tokenizer building" ON)
option(OCOS_ENABLE_TRIE_TOKENIZER "Enable the TrieTokenizer building" ON)
option(OCOS_ENABLE_SPM_TOKENIZER "Enable the SentencePiece tokenizer building" ON)
option(OCOS_ENABLE_WORDPIECE_TOKENIZER "Enable the WordpieceTokenizer building" ON)
option(OCOS_ENABLE_BERT_TOKENIZER "Enable the BertTokenizer building" ON)
option(OCOS_ENABLE_BLINGFIRE "Enable operators depending on the Blingfire library" ON)
option(OCOS_ENABLE_MATH "Enable math tensor operators building" ON)
option(OCOS_ENABLE_DLIB "Enable operators like Inverse depending on DLIB" ON)
option(OCOS_ENABLE_VENDOR_IMAGE_CODECS "Enable and use vendor image codecs if supported over libpng & libjpeg" OFF)
option(OCOS_ENABLE_OPENCV_CODECS "Enable cv2 and vision operators that require opencv imgcodecs." OFF)
option(OCOS_ENABLE_CV2 "Enable the operators in `operators/cv2`" OFF)
option(OCOS_ENABLE_VISION "Enable the operators in `operators/vision`" ON)
option(OCOS_ENABLE_AUDIO "Enable the operators for audio processing" ON)
option(OCOS_ENABLE_AZURE "Enable the operators for azure execution provider" OFF)

option(OCOS_ENABLE_STATIC_LIB "Enable generating static library" OFF)
option(OCOS_ENABLE_SELECTED_OPLIST "Enable including the selected_ops tool file" OFF)
option(OCOS_ENABLE_C_API "Enable building the C API" OFF)

option(OCOS_BUILD_SHARED_LIB "Enable building the dynamic library" ON)
option(OCOS_BUILD_PYTHON "Enable building the Python package" OFF)
option(OCOS_BUILD_JAVA "Enable building the Java package" OFF)
option(OCOS_BUILD_ANDROID "Enable building the Android package" OFF)
option(OCOS_BUILD_APPLE_FRAMEWORK "Enable building of the MacOS/iOS framework" OFF)

option(OCOS_USE_CUDA "Build the CUDA kernels" OFF)

# Optional value. Some operators do not support old versions due to using the new custom operator interface
# and will be disabled if this value is set and the version is incompatible.
set(OCOS_ONNXRUNTIME_VERSION "" CACHE STRING
  "The version of ONNX Runtime being used in the build. Format is <major>.<minor>.<patch>. e.g. 1.15.1")
set(OCOS_ONNXRUNTIME_PKG_URI "" CACHE STRING
  "Specify the onnxruntime C++ shared library zip package path, like ./onnxruntime-win-x64-1.16.0.zip")
set(OCOS_BUILD_PRESET "" CACHE STRING
  "Specify the build preset cmake settings file path, like 'token_api_only' which includes ./cmake/presets/token_api_only.cmake")

# AzureOp can be enabled by environment varaible OCOS_ENABLE_AZURE == 1
if (DEFINED ENV{OCOS_ENABLE_AZURE})
  set(OCOS_ENABLE_AZURE ON CACHE INTERNAL "" FORCE)
  message(STATUS "=> AzureOp is enabled env variable.")
endif()

function(disable_all_operators)
  set(OCOS_ENABLE_RE2_REGEX OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_TF_STRING OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_WORDPIECE_TOKENIZER OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_GPT2_TOKENIZER OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_TRIE_TOKENIZER OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_SPM_TOKENIZER OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_BERT_TOKENIZER OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_BLINGFIRE OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_MATH OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_DLIB OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_OPENCV_CODECS OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_CV2 OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_VISION OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_AZURE OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_AUDIO OFF CACHE INTERNAL "" FORCE)
endfunction()

if (CMAKE_GENERATOR_PLATFORM)
  # Multi-platform generator
  set(ocos_target_platform ${CMAKE_GENERATOR_PLATFORM})
else()
  set(ocos_target_platform ${CMAKE_SYSTEM_PROCESSOR})
endif()

if(NOT CC_OPTIMIZE)
  message(WARNING "!!!THE COMPILER OPTIMIZATION HAS BEEN DISABLED, DEBUG-ONLY!!!")
  string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
  string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  string(REGEX REPLACE "([\-\/]O[123])" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

  if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Od")
  endif()
endif()

if (MSVC)
  check_cxx_compiler_flag(-sdl HAS_SDL)
  check_cxx_compiler_flag(-Qspectre HAS_QSPECTRE)
  if (HAS_QSPECTRE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Qspectre")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qspectre")
  endif()
  set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /DYNAMICBASE")
  check_cxx_compiler_flag(-guard:cf HAS_GUARD_CF)
  if (HAS_GUARD_CF)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /guard:cf")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /guard:cf")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /guard:cf")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /guard:cf")
    set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} /guard:cf")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /guard:cf")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /guard:cf")
  endif()

  if (CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreaded" OR
      CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreadedDebug")
    set(_STATIC_MSVC_RUNTIME_LIBRARY ON)
  else()
    set(_STATIC_MSVC_RUNTIME_LIBRARY OFF)
  endif()
  message(STATUS "_STATIC_MSVC_RUNTIME_LIBRARY: ${_STATIC_MSVC_RUNTIME_LIBRARY}")

  # DLL initialization errors due to old conda msvcp140.dll dll are a result of the new MSVC compiler
  # See https://developercommunity.visualstudio.com/t/Access-violation-with-std::mutex::lock-a/10664660#T-N10668856
  # Remove this definition once the conda msvcp140.dll dll is updated.
  add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

if(NOT OCOS_BUILD_PYTHON AND OCOS_ENABLE_PYTHON)
  message("OCOS_ENABLE_PYTHON IS DEPRECATED, USE OCOS_BUILD_PYTHON INSTEAD")
  set(OCOS_BUILD_PYTHON ON CACHE INTERNAL "")
endif()

if(OCOS_BUILD_ANDROID)
  if(NOT CMAKE_TOOLCHAIN_FILE MATCHES "android.toolchain.cmake")
    message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE must be set to build/cmake/android.toolchain.cmake from the Android NDK.")
  endif()
  if(NOT ANDROID_PLATFORM OR NOT ANDROID_ABI)
    message(FATAL_ERROR "The Android platform (ANDROID_PLATFORM) and ABI (ANDROID_ABI) must be specified.")
  endif()

  set(OCOS_BUILD_JAVA ON CACHE INTERNAL "")
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/externals ${PROJECT_SOURCE_DIR}/cmake)

if(NOT PROJECT_IS_TOP_LEVEL AND ONNXRUNTIME_ROOT)
  set(_ONNXRUNTIME_EMBEDDED TRUE)
endif()

if (OCOS_ENABLE_SELECTED_OPLIST OR OCOS_BUILD_PRESET)
  disable_all_operators()
  if(OCOS_ENABLE_SELECTED_OPLIST)
    # Need to ensure _selectedoplist.cmake file is already generated in folder: ${PROJECT_SOURCE_DIR}/cmake/
    # You could run gen_selectedops.py in folder: tools/ to generate _selectedoplist.cmake
    message(STATUS "Looking for the _selectedoplist.cmake")
    include(_selectedoplist)
    # Include the selected_ops case, no way to run the unit tests, so disable it,
    # even the user explicitly set it to ON. (it is rare, most of the time, it is set by default)
    set(OCOS_ENABLE_CTEST OFF CACHE BOOL "" FORCE)
  endif()
  if (OCOS_BUILD_PRESET)
    set(_BUILD_PRESET "${PROJECT_SOURCE_DIR}/cmake/presets/${OCOS_BUILD_PRESET}.cmake")
    if (EXISTS ${_BUILD_PRESET})
      include(${_BUILD_PRESET})
    else()
      message(FATAL_ERROR "The specified build preset file does not exist: ${_BUILD_PRESET}")
    endif()
  endif()
endif()

set(_OCOS_EXCEPTIONS_REQUIRED OFF)
if (OCOS_ENABLE_GPT2_TOKENIZER OR
    OCOS_ENABLE_WORDPIECE_TOKENIZER OR
    OCOS_ENABLE_BLINGFIRE OR
    OCOS_ENABLE_SPM_TOKENIZER OR
    (OCOS_ENABLE_CV2 OR OCOS_ENABLE_OPENCV_CODECS OR OCOS_ENABLE_VISION))
    set(_OCOS_EXCEPTIONS_REQUIRED ON)
endif()

# Special case an embedded build with ORT exceptions disabled but custom ops that require exceptions.
# Allow using an override so we can do a direct build of ort-ext in a CI without having to embed it in an ORT build.
set(_OCOS_PREVENT_EXCEPTION_PROPAGATION OFF)
if (_OCOS_PREVENT_EXCEPTION_PROPAGATION_OVERRIDE)
  set(_OCOS_PREVENT_EXCEPTION_PROPAGATION ${_OCOS_PREVENT_EXCEPTION_PROPAGATION_OVERRIDE})
elseif(_ONNXRUNTIME_EMBEDDED AND onnxruntime_DISABLE_EXCEPTIONS AND _OCOS_EXCEPTIONS_REQUIRED)
  set(_OCOS_PREVENT_EXCEPTION_PROPAGATION ON)
endif()

if (_OCOS_PREVENT_EXCEPTION_PROPAGATION)
  message(STATUS "Embedded build as part of ONNX Runtime with exceptions disabled. "
                 "Extensions will be built with exceptions enabled due to included custom ops "
                 "using 3rd party libraries that require exceptions.")

  if (NOT OCOS_ENABLE_CPP_EXCEPTIONS)
    message(WARNING "Enabling C++ exception support as custom ops included in the build require them to be enabled.")
    set(OCOS_ENABLE_CPP_EXCEPTIONS ON)
  endif()

  # undo the flags that ORT has set to disable exceptions.
  # see https://github.com/microsoft/onnxruntime/blob/b1abb8c656c597bf221bd85682ae3d9e350d9aba/cmake/adjust_global_compile_flags.cmake#L160-L169
  if(MSVC)
    string(REPLACE "/EHs-c-" "/EHsc" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    string(REPLACE "-fno-exceptions" "-fexceptions" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-fno-unwind-tables" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-fno-asynchronous-unwind-tables" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  endif()

  # the ort-ext code has to provide a barrier between the exception enabled custom op code and ORT.
  add_compile_definitions(OCOS_PREVENT_EXCEPTION_PROPAGATION)
endif()

if(NOT OCOS_ENABLE_CPP_EXCEPTIONS)
  add_compile_definitions(OCOS_NO_EXCEPTIONS ORT_NO_EXCEPTIONS)
  if (NOT _ONNXRUNTIME_EMBEDDED)
    add_compile_definitions(_HAS_EXCEPTIONS=0)
  endif()
endif()

include(FetchContent)

function(set_msvc_c_cpp_compiler_warning_level warning_level)
  if (NOT "${warning_level}" MATCHES "^[0-4]$")
    message(FATAL_ERROR "Expected warning_level value of 0-4, got '${warning_level}'.")
  endif()

  if (MSVC)
    set(warning_flag "/W${warning_level}")
    get_property(opts DIRECTORY PROPERTY COMPILE_OPTIONS)
    # only match the generator expression added by this function
    list(FILTER opts
         EXCLUDE REGEX "^\\$<\\$<OR:\\$<COMPILE_LANGUAGE:C>,\\$<COMPILE_LANGUAGE:CXX>>:/W[0-4]>$")
    list(APPEND opts "$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${warning_flag}>")
    set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "${opts}")
  endif()
endfunction()

if (NOT ONNXRUNTIME_INCLUDE_DIR)
  include(ext_ortlib)
endif()
# set default MSVC warning level to 3 for external dependencies
set_msvc_c_cpp_compiler_warning_level(3)
include(gsl)

macro(standardize_output_folder bin_target)
  set_target_properties(${bin_target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endmacro()

if(OCOS_USE_CUDA)
  include(ext_cuda)
  include(cutlass)
endif()

#######################################################################################################################
# build the operator file list from the build flag.

if(OCOS_ENABLE_RE2_REGEX)
  if(NOT TARGET re2::re2)
    set(RE2_BUILD_TESTING OFF CACHE INTERNAL "")
    message(STATUS "Fetch googlere2")
    include(googlere2)
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set_property(TARGET re2 PROPERTY COMPILE_OPTIONS)
  endif()
endif()

# ### scan all source files
file(GLOB TARGET_SRC_NOEXCEPTION "base/*.h" "base/*.cc")
file(GLOB TARGET_SRC "operators/*.cc" "operators/*.h" "include/*.h" "include/*.hpp")

if(OCOS_ENABLE_DLIB)
  set(DLIB_ISO_CPP_ONLY ON CACHE INTERNAL "")
  set(DLIB_NO_GUI_SUPPORT ON CACHE INTERNAL "")
  set(DLIB_USE_CUDA OFF CACHE INTERNAL "")
  set(DLIB_USE_LAPACK OFF CACHE INTERNAL "")
  set(DLIB_USE_BLAS OFF CACHE INTERNAL "")
  include(dlib)

  # Ideally, dlib should be included as file(GLOB TARGET_SRC_DLIB "${dlib_SOURCE_DIR}/dlib/all/source.cpp")
  # To avoid the unintentional using some unwanted component, we only include the test_for_odr_violations.cpp
  # to check if there is any violation in build configuration to ensure compiling some dlib source files correctly.
  file(GLOB TARGET_SRC_DLIB "${dlib_SOURCE_DIR}/dlib/test_for_odr_violations.cpp")
endif()

if(OCOS_ENABLE_TF_STRING)
  set(farmhash_SOURCE_DIR ${PROJECT_SOURCE_DIR}/cmake/externals/farmhash)
  file(GLOB TARGET_SRC_KERNELS "operators/text/*.cc" "operators/text/*.h*")
  file(GLOB TARGET_SRC_HASH "${farmhash_SOURCE_DIR}/src/farmhash.*")
  list(APPEND TARGET_SRC_NOEXCEPTION ${TARGET_SRC_KERNELS} ${TARGET_SRC_HASH})
endif()

if(OCOS_ENABLE_AUDIO)
  if (NOT OCOS_ENABLE_DLIB)
    message(FATAL_ERROR "Audio operators require DLIB to be enabled.")
  endif()
  include(dr_libs)
  file(GLOB TARGET_SRC_AUDIO "operators/audio/*.*")
  list(APPEND TARGET_SRC_AUDIO ${dlib_SOURCE_DIR}/dlib/fft/fft.cpp)
  list(APPEND TARGET_SRC_NOEXCEPTION ${TARGET_SRC_AUDIO})
endif()

if(OCOS_ENABLE_RE2_REGEX)
  file(GLOB TARGET_SRC_RE2_KERNELS "operators/text/re2_strings/*.cc" "operators/text/re2_strings/*.h*")
  list(APPEND TARGET_SRC_NOEXCEPTION ${TARGET_SRC_RE2_KERNELS})
endif()

if(OCOS_ENABLE_MATH)
  if (NOT OCOS_ENABLE_DLIB)
    message(FATAL_ERROR "Math operators require DLIB to be enabled.")
  endif()
  file(GLOB TARGET_SRC_INVERSE "operators/math/dlib/*.cc" "operators/math/dlib/*.h*")

  file(GLOB TARGET_SRC_MATH "operators/math/*.cc" "operators/math/*.h*")
  if(OCOS_USE_CUDA)
    file(GLOB TARGET_SRC_MATH_CUDA "operators/math/cuda/*.*")
    list(APPEND TARGET_SRC_MATH ${TARGET_SRC_MATH_CUDA})
  endif()

  list(APPEND TARGET_SRC ${TARGET_SRC_MATH} ${TARGET_SRC_DLIB} ${TARGET_SRC_INVERSE})
endif()

if (OCOS_USE_CUDA)
  file(GLOB_RECURSE TARGET_SRC_CUDA "operators/cuda/*.*")
  list(APPEND TARGET_SRC ${TARGET_SRC_CUDA})
endif()

# enable the opencv dependency if we have ops that require it
if(OCOS_ENABLE_CV2)
  set(_ENABLE_OPENCV ON)
  message(STATUS "Fetch opencv")
  include(opencv)
endif()

if(OCOS_ENABLE_CV2)
  file(GLOB TARGET_SRC_CV2 "operators/cv2/*.cc" "operators/cv2/*.h*")
  file(GLOB TARGET_SRC_CV2_IMGPROC_OPS "operators/cv2/imgproc/*.cc" "operators/cv2/imgproc/*.h*")
  file(GLOB TARGET_SRC_CV2_IMGCODECS_OPS "operators/cv2/imgcodecs/*.cc" "operators/cv2/imgcodecs/*.h*")

  list(APPEND TARGET_SRC ${TARGET_SRC_CV2})
  list(APPEND TARGET_SRC ${TARGET_SRC_CV2_IMGPROC_OPS})
  if (OCOS_ENABLE_OPENCV_CODECS)
    list(APPEND TARGET_SRC ${TARGET_SRC_CV2_IMGCODECS_OPS})
  endif()
endif()

if(OCOS_ENABLE_VISION)
  file(GLOB TARGET_SRC_VISION "operators/vision/*.cc" "operators/vision/*.h*")
  list(APPEND TARGET_SRC ${TARGET_SRC_VISION})
endif()

set(_HAS_TOKENIZER OFF)

if(OCOS_ENABLE_GPT2_TOKENIZER)
  # GPT2
  set(_HAS_TOKENIZER ON)
  file(GLOB tok_TARGET_SRC "operators/tokenizer/bpe_*.*" "operators/tokenizer/unicode*.*")
  list(APPEND TARGET_SRC ${tok_TARGET_SRC})
endif()

if(OCOS_ENABLE_TRIE_TOKENIZER)
  # Trie Tokenizer
  set(_HAS_TOKENIZER ON)
  file(GLOB tok_TARGET_SRC "operators/tokenizer/trie_tokenizer.hpp" "operators/tokenizer/unescape.h")
  list(APPEND TARGET_SRC ${tok_TARGET_SRC})
endif()

if(OCOS_ENABLE_SPM_TOKENIZER)
  # SentencePiece
  set(_HAS_TOKENIZER ON)
  set(SPM_ENABLE_TCMALLOC OFF CACHE INTERNAL "")
  set(SPM_ENABLE_SHARED OFF CACHE INTERNAL "")
  message(STATUS "Fetch sentencepiece")
  include(sentencepieceproject)
  file(GLOB stpiece_TARGET_SRC "operators/tokenizer/sentencepiece/*.cc" "operators/tokenizer/sentencepiece*")
  list(REMOVE_ITEM stpiece_TARGET_SRC INCLUDE REGEX ".*((spm)|(train)).*")
  list(APPEND TARGET_SRC ${stpiece_TARGET_SRC})
endif()

if(OCOS_ENABLE_WORDPIECE_TOKENIZER)
  set(_HAS_TOKENIZER ON)
  file(GLOB wordpiece_TARGET_SRC "operators/tokenizer/wordpiece*.*")
  list(APPEND TARGET_SRC ${wordpiece_TARGET_SRC})
endif()

if(OCOS_ENABLE_BERT_TOKENIZER)
  # Bert
  set(_HAS_TOKENIZER ON)
  file(GLOB bert_TARGET_SRC "operators/tokenizer/basic_tokenizer.*"
                            "operators/tokenizer/bert_tokenizer.*"
                            "operators/tokenizer/bert_tokenizer_decoder.*")
  list(APPEND TARGET_SRC ${bert_TARGET_SRC})
endif()

if(OCOS_ENABLE_BLINGFIRE)
  # blingfire
  set(_HAS_TOKENIZER ON)
  file(GLOB blingfire_TARGET_SRC "operators/tokenizer/blingfire*.*")
  list(APPEND TARGET_SRC ${blingfire_TARGET_SRC})
endif()

# Build with Azure ops is disabled in a number of places due to the following reasons:
# - not setup for macOS, iOS or WASM
# - link error on Windows 32-bit ARM from some functions in wincrpt.h (e.g. CryptDestroyHash)
#   - not sure if they're available on ARM 32.
#   - we explicitly add crypt32 to the link list which is where they should be defined.
# - build with static CRT on Windows causes link errors with triton. may not be supported.
# - x86 Android build has error when using static openssl.
#   - TODO: need to investigate further. currently getting load error with dynamic libs and runtime error with static libs
if(OCOS_ENABLE_AZURE)
  if (APPLE OR
      CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR
      (MSVC AND CMAKE_GENERATOR_PLATFORM STREQUAL "ARM") OR
      (MSVC AND _STATIC_MSVC_RUNTIME_LIBRARY) OR
      (ANDROID AND ANDROID_ABI STREQUAL "x86"))
    message(STATUS "Excluding Azure custom operators as they are not currently supported in this type of build. ")
    set(OCOS_ENABLE_AZURE OFF)

  elseif(OCOS_ONNXRUNTIME_VERSION)
    string(REPLACE "." ";" VERSION_LIST ${OCOS_ONNXRUNTIME_VERSION})
    list(GET VERSION_LIST 0 ORT_VERSION_MAJOR)
    list(GET VERSION_LIST 1 ORT_VERSION_MINOR)
    if(ORT_VERSION_MAJOR EQUAL 1 AND ORT_VERSION_MINOR LESS 14)
      message(STATUS "Azure custom operators are not supported with ONNX Runtime versions < 1.14. "
                     "Excluding them from the build.")
      set(OCOS_ENABLE_AZURE OFF)
    endif()
  endif()
endif()

if(OCOS_ENABLE_AZURE)
  # Azure endpoint invokers
  file(GLOB TARGET_SRC_AZURE "operators/azure/*.cc" "operators/azure/*.h*")

  if (ANDROID)
    include(curl)

    # make sure these work
    find_package(OpenSSL REQUIRED)
    find_package(CURL REQUIRED)

    # exclude triton as it's not supported on Android
    list(FILTER TARGET_SRC_AZURE EXCLUDE REGEX ".*triton.*")
  else()
    add_compile_definitions(AZURE_INVOKERS_ENABLE_TRITON)

    # need to set the correct vcpkg target platform strings before including triton
    if (WIN32)
      set(vcpkg_target_platform ${ocos_target_platform})

      # remap known variants
      if (ocos_target_platform STREQUAL "AMD64")
        set(vcpkg_target_platform "x64")
      elseif(ocos_target_platform STREQUAL "Win32")
        set(vcpkg_target_platform "x86")
      elseif(ocos_target_platform STREQUAL "ARM64")
        set(vcpkg_target_platform "arm64")
      elseif(ocos_target_platform STREQUAL "ARM")
        set(vcpkg_target_platform "arm")
      endif()

      if(NOT "${vcpkg_target_platform}" MATCHES "^(x64|x86|arm64|arm)$")
        message(FATAL_ERROR "Unexpected ocos_target_platform: ${ocos_target_platform}")
      endif()

      # vcpkg is used for the azure ops. we build static versions so they're directly included in the extensions library
      # in the python package.
      set(vcpkg_triplet ${vcpkg_target_platform}-windows-static-md)

      message(STATUS "vcpkg_triplet: ${vcpkg_triplet}")
    endif()

    include(triton)
  endif()

  list(APPEND TARGET_SRC ${TARGET_SRC_AZURE})
endif()

if(OCOS_ENABLE_GPT2_TOKENIZER OR OCOS_ENABLE_WORDPIECE_TOKENIZER)
  message(STATUS "Fetch json")
  include(json)
endif()

if(_HAS_TOKENIZER)
  message(STATUS "Tokenizer needed.")
  file(GLOB tokenizer_TARGET_SRC "operators/tokenizer/tokenizers.*" "operators/tokenizer/*.hpp")
  list(APPEND TARGET_SRC ${tokenizer_TARGET_SRC})
endif()

### make all compile options.
if(MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>" "$<$<COMPILE_LANGUAGE:CXX,C>:/utf-8>")
endif()

# Library will not be built if the target src for the lib does not contain any valid *.cc files.
# Hence the placeholders `noexcep_operators_placeholder.cc` and `ocos_operators_placeholder.cc`
# are added to the target sources for TARGET_SRC_NOEXCEPTION and TARGET_SRC respectively
# to ensure the libraries get built.
add_library(noexcep_operators STATIC ${TARGET_SRC_NOEXCEPTION})
add_library(ocos_operators STATIC ${TARGET_SRC})
# TODO: need to address the SDL warnings happens on custom operator code.
# if (HAS_SDL)
#   target_compile_options(ocos_operators PRIVATE "/sdl")
# endif()
set_target_properties(noexcep_operators PROPERTIES FOLDER "operators")
set_target_properties(ocos_operators PROPERTIES FOLDER "operators")

# filter out any files in ${TARGET_SRC} which don't have prefix of ${PROJECT_SOURCE_DIR} before calling source_group
set(_TARGET_SRC_FOR_SOURCE_GROUP)
foreach(_TARGET_SRC_FILE IN LISTS TARGET_SRC)
  cmake_path(IS_PREFIX PROJECT_SOURCE_DIR ${_TARGET_SRC_FILE}
             NORMALIZE
             _is_prefix_result)
  if(_is_prefix_result)
    list(APPEND _TARGET_SRC_FOR_SOURCE_GROUP ${_TARGET_SRC_FILE})
  endif()
endforeach()
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${_TARGET_SRC_FOR_SOURCE_GROUP})

standardize_output_folder(noexcep_operators)
standardize_output_folder(ocos_operators)

target_include_directories(noexcep_operators PUBLIC
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${GSL_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/custom_op
  ${PROJECT_SOURCE_DIR}/base
  ${PROJECT_SOURCE_DIR}/operators)

target_include_directories(ocos_operators PUBLIC
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${GSL_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/custom_op
  ${PROJECT_SOURCE_DIR}/base
  ${PROJECT_SOURCE_DIR}/operators)

if (OCOS_USE_CUDA)
  target_include_directories(ocos_operators PUBLIC ${cutlass_SOURCE_DIR}/include ${cutlass_SOURCE_DIR}/examples)
endif()

set(ocos_libraries)
set(OCOS_COMPILE_DEFINITIONS)

if(OCOS_ENABLE_DLIB)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_DLIB)
endif()

if (OCOS_ENABLE_AUDIO)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_DR_LIBS)
  target_include_directories(noexcep_operators PUBLIC ${dr_libs_SOURCE_DIR})
endif()

if(_HAS_TOKENIZER)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_TOKENIZER)
  target_include_directories(ocos_operators PUBLIC
    ${PROJECT_SOURCE_DIR}/operators/tokenizer)
endif()

if(OCOS_ENABLE_TF_STRING)
  target_include_directories(noexcep_operators PUBLIC ${farmhash_SOURCE_DIR}/src)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_TF_STRING NOMINMAX FARMHASH_NO_BUILTIN_EXPECT FARMHASH_DEBUG=0)

  if(OCOS_ENABLE_RE2_REGEX)
    target_include_directories(noexcep_operators PUBLIC ${googlere2_SOURCE_DIR})
    target_link_libraries(noexcep_operators PRIVATE re2)
  endif()
endif()

if(OCOS_ENABLE_RE2_REGEX)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_RE2_REGEX)
endif()

if(OCOS_ENABLE_MATH)
  target_include_directories(ocos_operators PUBLIC ${dlib_SOURCE_DIR})
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_MATH)
endif()

if(_ENABLE_OPENCV)
  list(APPEND ocos_libraries ${opencv_LIBS})
  target_include_directories(ocos_operators PUBLIC ${opencv_INCLUDE_DIRS})
endif()

if(OCOS_ENABLE_OPENCV_CODECS)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_OPENCV_CODECS)
endif()

if(OCOS_ENABLE_CV2)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_CV2)
endif()

if(OCOS_ENABLE_VISION)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_VISION)
  set(_DEFAULT_CODEC_ENABLE ON)
  if(OCOS_ENABLE_VENDOR_IMAGE_CODECS)
    add_compile_definitions(OCOS_ENABLE_VENDOR_IMAGE_CODECS)
    if(WIN32)
      # Use WIC on Windows. Nothing to be done
      set(_DEFAULT_CODEC_ENABLE OFF)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
      # Use ImageIO on Apple platforms
      set(_DEFAULT_CODEC_ENABLE OFF)
      target_link_libraries(ocos_operators PRIVATE "-framework CoreFoundation" "-framework CoreGraphics" "-framework ImageIO")
    endif()
  endif()

  set(_DEFAULT_CODEC_ENABLE ON) # libpng and libjpeg can be optional after EncodeImage with native support too.
  if(_DEFAULT_CODEC_ENABLE)
    if (NOT OCOS_ENABLE_DLIB)
      message(FATAL_ERROR "Vision operators require DLIB to be enabled.") # for now, we need dlib for image processing
    endif()
    include(ext_imgcodecs)  
    target_include_directories(ocos_operators PUBLIC ${libPNG_SOURCE_DIR} ${libJPEG_SOURCE_DIR})
    target_link_libraries(ocos_operators PUBLIC ${PNG_LIBRARY} ${JPEG_LIBRARY})
  endif()
endif()

if(OCOS_ENABLE_AZURE)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_AZURE)
endif()

if (OCOS_ENABLE_C_API)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_C_API)
endif()

if(OCOS_ENABLE_GPT2_TOKENIZER)
  # GPT2
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_GPT2_TOKENIZER)
endif()

if(OCOS_ENABLE_TRIE_TOKENIZER)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_TRIE_TOKENIZER)
endif()

if(OCOS_ENABLE_WORDPIECE_TOKENIZER)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_WORDPIECE_TOKENIZER)
endif()

if(OCOS_ENABLE_BERT_TOKENIZER)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_BERT_TOKENIZER)
endif()

if(OCOS_ENABLE_SPM_TOKENIZER)
  # SentencePiece
  target_include_directories(ocos_operators PUBLIC ${spm_INCLUDE_DIRS})
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_SPM_TOKENIZER)
  list(APPEND ocos_libraries sentencepiece-static)
endif()

if(OCOS_ENABLE_BLINGFIRE)
  include(blingfire)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_BLINGFIRE)
  list(APPEND ocos_libraries bingfirtinydll_static)
endif()

if(OCOS_ENABLE_GPT2_TOKENIZER OR OCOS_ENABLE_WORDPIECE_TOKENIZER)
  target_include_directories(ocos_operators PUBLIC ${nlohmann_json_SOURCE_DIR}/include)
endif()

# If building a shared library we can't throw an internal exception type across the library boundary as the type
# will be unknown. Set a compile definition so the code can adjust to the build type.
if(OCOS_BUILD_SHARED_LIB)
  list(APPEND OCOS_COMPILE_DEFINITIONS OCOS_SHARED_LIBRARY)
endif()

# __android_log_print support
if(ANDROID)
  list(APPEND ocos_libraries log)
endif()

list(REMOVE_DUPLICATES OCOS_COMPILE_DEFINITIONS)
target_compile_definitions(noexcep_operators PRIVATE ${OCOS_COMPILE_DEFINITIONS})
if(NOT OCOS_ENABLE_CPP_EXCEPTIONS)
  if(MSVC)
    get_target_property(_target_cxx_flags noexcep_operators COMPILE_OPTIONS)
    list(REMOVE_ITEM _target_cxx_flags "/EHsc")
    list(APPEND _target_cxx_flags "/EHs-c-")
    set_target_properties(noexcep_operators PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
  else()
    target_compile_options(noexcep_operators PRIVATE -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables)
  endif()
endif()

list(APPEND ocos_libraries noexcep_operators)
target_compile_definitions(ocos_operators PRIVATE ${OCOS_COMPILE_DEFINITIONS})
target_link_libraries(ocos_operators PRIVATE ${ocos_libraries})

file(GLOB _TARGET_LIB_SRC "shared/lib/*.cc")
if(OCOS_ENABLE_C_API)
  file(GLOB utils_TARGET_SRC "shared/api/c_api_utils.*" "shared/api/runner.hpp")
  list(APPEND _TARGET_LIB_SRC ${utils_TARGET_SRC})
  if(_HAS_TOKENIZER)
    file(GLOB tok_TARGET_SRC "shared/api/c_api_tokenizer.cc" "shared/api/token*")
    list(APPEND _TARGET_LIB_SRC ${tok_TARGET_SRC})
  endif()
  if(OCOS_ENABLE_AUDIO)
    file(GLOB audio_TARGET_SRC "shared/api/c_api_feature_extraction.*" "shared/api/speech_*")
    list(APPEND _TARGET_LIB_SRC ${audio_TARGET_SRC})
  endif()
  if(OCOS_ENABLE_DLIB)
    file(GLOB cv2_TARGET_SRC "shared/api/c_api_processor.*" "shared/api/image_*.*")
    list(APPEND _TARGET_LIB_SRC ${cv2_TARGET_SRC})
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  if(OCOS_ENABLE_STATIC_LIB)
    message(FATAL_ERROR "Emscripten build does not support building a static library.")
  endif()
  # Emscripten does not support building a shared library with custom ops.
  # and backward compatible with the previous version, we silently turn off the shared library build.
  set(OCOS_BUILD_SHARED_LIB OFF CACHE INTERNAL "" FORCE)
  add_executable(ortcustomops ${_TARGET_LIB_SRC})
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

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_property(TARGET ortcustomops APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=1 -s DEMANGLE_SUPPORT=1")
  else()
    set_property(TARGET ortcustomops APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=0 -s DEMANGLE_SUPPORT=0")
  endif()
else()
  add_library(ortcustomops STATIC ${_TARGET_LIB_SRC})
  if (HAS_SDL)
    target_compile_options(ortcustomops PRIVATE "/sdl")
  endif()
  add_library(onnxruntime_extensions ALIAS ortcustomops)
  standardize_output_folder(ortcustomops)
endif()
set_target_properties(ortcustomops PROPERTIES FOLDER "operators")

if(OCOS_ENABLE_AZURE)
  if (ANDROID)
    # find_package calls were made immediately after `include(curl)` so we know CURL and OpenSSL are available.
    find_package(ZLIB REQUIRED)
    target_link_libraries(ocos_operators PUBLIC CURL::libcurl OpenSSL::Crypto OpenSSL::SSL ZLIB::ZLIB)
  else()
    # we need files from the triton client (e.g. curl header on linux) to be available for the ocos_operators build.
    # add a dependency so the fetch and build of triton happens first.
    add_dependencies(ocos_operators triton)
    target_include_directories(ocos_operators PUBLIC ${triton_INSTALL_DIR}/include)

    target_link_directories(ocos_operators PUBLIC ${triton_INSTALL_DIR}/lib)
    target_link_directories(ocos_operators PUBLIC ${triton_INSTALL_DIR}/lib64)

    if (WIN32)
      # As per https://curl.se/docs/faq.html#Link_errors_when_building_libcur we need to set CURL_STATICLIB.
      target_compile_definitions(ocos_operators PRIVATE CURL_STATICLIB)
      target_include_directories(ocos_operators PUBLIC ${VCPKG_SRC}/installed/${vcpkg_triplet}/include)

      if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(curl_LIB_NAME "libcurl-d")
        set(zlib_LIB_NAME "zlibd")
        target_link_directories(ocos_operators PUBLIC ${VCPKG_SRC}/installed/${vcpkg_triplet}/debug/lib)
      else()
        set(curl_LIB_NAME "libcurl")
        set(zlib_LIB_NAME "zlib")
        target_link_directories(ocos_operators PUBLIC ${VCPKG_SRC}/installed/${vcpkg_triplet}/lib)
      endif()
      target_link_libraries(ocos_operators PUBLIC httpclient_static ${curl_LIB_NAME} libcrypto libssl ${zlib_LIB_NAME} ws2_32 crypt32 Wldap32)
    else()
      find_package(ZLIB REQUIRED)

      # If finding the OpenSSL or CURL package fails for a local build you can install libcurl4-openssl-dev.
      # See also info on triton client dependencies here: https://github.com/triton-inference-server/client
      find_package(OpenSSL REQUIRED)
      find_package(CURL)

      if (CURL_FOUND)
        message(STATUS "Found CURL package")
        set(libcurl_target CURL::libcurl)
      else()
        # curl is coming from triton but as that's an external project it isn't built yet and we have to add
        # paths and library names instead of cmake targets.
        message(STATUS "Using CURL build from triton client. Once built it should be in ${triton_THIRD_PARTY_DIR}/curl")
        target_include_directories(ocos_operators PUBLIC ${triton_THIRD_PARTY_DIR}/curl/include)

        # Install is to 'lib' except on CentOS (which is used for the manylinux build of the python wheel).
        # Side note: we have to patch the triton client CMakeLists.txt to only use 'lib64' for 64-bit builds otherwise
        #            the build of the 32-bit python wheel fails with CURL not being found due to the invalid library
        #            directory name.
        target_link_directories(ocos_operators PUBLIC ${triton_THIRD_PARTY_DIR}/curl/lib)
        target_link_directories(ocos_operators PUBLIC ${triton_THIRD_PARTY_DIR}/curl/lib64)

        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
          set(libcurl_target "curl-d")
        else()
          set(libcurl_target "curl")
        endif()
      endif()

      target_link_libraries(ocos_operators PUBLIC httpclient_static ${libcurl_target} OpenSSL::Crypto OpenSSL::SSL ZLIB::ZLIB)
    endif()
  endif()
endif()

target_compile_definitions(ortcustomops PUBLIC ${OCOS_COMPILE_DEFINITIONS})
target_include_directories(ortcustomops PUBLIC "$<TARGET_PROPERTY:noexcep_operators,INTERFACE_INCLUDE_DIRECTORIES>")
target_include_directories(ortcustomops PUBLIC "$<TARGET_PROPERTY:ocos_operators,INTERFACE_INCLUDE_DIRECTORIES>")

target_link_libraries(ortcustomops PUBLIC ocos_operators)

if(OCOS_BUILD_SHARED_LIB)
  file(GLOB shared_TARGET_SRC "shared/*.cc" "shared/*.h")
  if (OCOS_ENABLE_C_API)
    if (NOT _HAS_TOKENIZER OR NOT OCOS_ENABLE_AUDIO)
      message(FATAL_ERROR "Shared library build requires GPT2_TOKENIZER, AUDIO to be enabled.")
    endif()
    list(APPEND shared_TARGET_SRC "shared/extensions_c.def")
  else()
    list(APPEND shared_TARGET_SRC "shared/ortcustomops.def")
  endif()
  add_library(extensions_shared SHARED ${shared_TARGET_SRC})

  # We need to propagate OCOS_SHARED_LIBRARY if set.
  # could specifically add that if using OCOS_COMPILE_DEFINITIONS is too much.
  target_compile_definitions(extensions_shared PRIVATE ${OCOS_COMPILE_DEFINITIONS})

  source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${shared_TARGET_SRC})
  standardize_output_folder(extensions_shared)

  if(LINUX OR ANDROID)
    set_property(TARGET extensions_shared APPEND_STRING PROPERTY LINK_FLAGS
      " -Wl,--version-script -Wl,${PROJECT_SOURCE_DIR}/shared/ortcustomops.ver")
    # strip if not a debug build
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
      set_property(TARGET extensions_shared APPEND_STRING PROPERTY LINK_FLAGS " -Wl,-s")
    endif()
  endif()

  target_include_directories(extensions_shared PUBLIC $<TARGET_PROPERTY:ortcustomops,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(extensions_shared PRIVATE ortcustomops)
  set_target_properties(extensions_shared PROPERTIES OUTPUT_NAME "ortextensions")
  if(MSVC AND ocos_target_platform MATCHES "x86|x64")
    target_link_options(extensions_shared PRIVATE "/CETCOMPAT")
  endif()

  # Set some properties of your target
  set_target_properties(extensions_shared PROPERTIES
    VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}
    SOVERSION ${CPACK_PACKAGE_VERSION_MAJOR}
  )

  # Install your target
  install(TARGETS extensions_shared
    RUNTIME DESTINATION bin
  )

endif()

if(OCOS_USE_CUDA)
  target_include_directories(ocos_operators PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_include_directories(ortcustomops PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(extensions_shared PUBLIC cudart cublas cufft)
endif()

if(OCOS_BUILD_PYTHON)
  message(STATUS "Python Build is enabled")
  set(shared_TARGET_LIB_SRC ${_TARGET_LIB_SRC}) # these library file are also needed for python build
  include(ext_python)
endif()

if(OCOS_BUILD_JAVA)
  message(STATUS "Java Build is enabled")
  include(ext_java)
endif()

if(OCOS_BUILD_APPLE_FRAMEWORK)
  include(ext_apple_framework)
  if (MAC_CATALYST)
    add_compile_options(-Wno-overriding-t-option)
    add_link_options(-Wno-overriding-t-option)
  endif()
endif()

if (_ORTX_STANDALONE_PROJECT)
  # clean up the requirements.txt files from 3rd party project folder to suppress the code security false alarms
  file(GLOB_RECURSE NO_USE_FILES ${CMAKE_BINARY_DIR}/_deps/*requirements.txt)
  message(STATUS "Found the following requirements.txt: ${NO_USE_FILES}")

  foreach(nf ${NO_USE_FILES})
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${nf})
  endforeach()

  # Run CPack to generate the NuGet package
  include(CPack)

  if(OCOS_ENABLE_CTEST AND NOT MAC_CATALYST)
    include(ext_tests)
  endif()
endif()
