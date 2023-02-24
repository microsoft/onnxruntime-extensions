cmake_minimum_required(VERSION 3.20)
project(onnxruntime_extensions LANGUAGES C CXX)

# set(CMAKE_VERBOSE_MAKEFILE ON)
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo." FORCE)
endif()

function(read_version_file major_var minor_var patch_var)
  set(version_file ${PROJECT_SOURCE_DIR}/version.txt)
  file(READ ${version_file} version_file_content)
  if(version_file_content MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
    set(${major_var} ${CMAKE_MATCH_1} PARENT_SCOPE)
    set(${minor_var} ${CMAKE_MATCH_2} PARENT_SCOPE)
    set(${patch_var} ${CMAKE_MATCH_3} PARENT_SCOPE)
  else()
    message(FATAL_ERROR "Failed to parse version from file: ${version_file}")
  endif()
endfunction()

set(CPACK_PACKAGE_NAME "onnxruntime_extensions")
read_version_file(CPACK_PACKAGE_VERSION_MAJOR CPACK_PACKAGE_VERSION_MINOR CPACK_PACKAGE_VERSION_PATCH)
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
  cmake_policy(SET CMP0077 NEW)
endif()

# Needed for Java
set(CMAKE_C_STANDARD 99)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CheckCXXCompilerFlag)
include(CheckLanguage)

option(CC_OPTIMIZE "Allow compiler optimizations, Set to OFF to disable" ON)
option(OCOS_ENABLE_PYTHON "Enable Python component building, (deprecated)" OFF)
option(OCOS_ENABLE_CTEST "Enable C++ test" OFF)
option(OCOS_ENABLE_CPP_EXCEPTIONS "Enable C++ Exception" ON)
option(OCOS_ENABLE_TF_STRING "Enable String Operator Set" ON)
option(OCOS_ENABLE_RE2_REGEX "Enable StringRegexReplace and StringRegexSplit" ON)
option(OCOS_ENABLE_GPT2_TOKENIZER "Enable the GPT2 tokenizer building" ON)
option(OCOS_ENABLE_SPM_TOKENIZER "Enable the SentencePiece tokenizer building" ON)
option(OCOS_ENABLE_WORDPIECE_TOKENIZER "Enable the WordpieceTokenizer building" ON)
option(OCOS_ENABLE_BERT_TOKENIZER "Enable the BertTokenizer building" ON)
option(OCOS_ENABLE_BLINGFIRE "Enable operators depending on the Blingfire library" ON)
option(OCOS_ENABLE_MATH "Enable math tensor operators building" ON)
option(OCOS_ENABLE_DLIB "Enable operators like Inverse depending on DLIB" ON)
option(OCOS_ENABLE_OPENCV_CODECS "Enable cv2 and vision operators that require opencv imgcodecs." ON)
option(OCOS_ENABLE_CV2 "Enable the operators in `operators/cv2`" ON)
option(OCOS_ENABLE_VISION "Enable the operators in `operators/vision`" ON)

option(OCOS_ENABLE_STATIC_LIB "Enable generating static library" OFF)
option(OCOS_ENABLE_SELECTED_OPLIST "Enable including the selected_ops tool file" OFF)
option(OCOS_BUILD_PYTHON "Enable building the Python package" OFF)
option(OCOS_BUILD_JAVA "Enable building the Java package" OFF)
option(OCOS_BUILD_ANDROID "Enable building the Android package" OFF)
option(OCOS_BUILD_APPLE_FRAMEWORK "Enable building of the MacOS/iOS framework" OFF)

function(disable_all_operators)
  set(OCOS_ENABLE_RE2_REGEX OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_TF_STRING OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_WORDPIECE_TOKENIZER OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_GPT2_TOKENIZER OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_SPM_TOKENIZER OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_BERT_TOKENIZER OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_BLINGFIRE OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_MATH OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_DLIB OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_OPENCV_CODECS OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_CV2 OFF CACHE INTERNAL "" FORCE)
  set(OCOS_ENABLE_VISION OFF CACHE INTERNAL "" FORCE)
endfunction()

if (CMAKE_GENERATOR_PLATFORM)
  # Multi-platform generator
  set(ocos_target_platform ${CMAKE_GENERATOR_PLATFORM})
else()
  set(ocos_target_platform ${CMAKE_SYSTEM_PROCESSOR})
endif()

if(NOT CC_OPTIMIZE)
  message("!!!THE COMPILER OPTIMIZATION HAS BEEN DISABLED, DEBUG-ONLY!!!")
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

set(CMAKE_FIND_FRAMEWORK NEVER CACHE STRING "...")

if(NOT "${CMAKE_FIND_FRAMEWORK}" STREQUAL "NEVER")
  message(FATAL_ERROR "CMAKE_FIND_FRAMEWORK is not NEVER")
endif()

# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/externals ${PROJECT_SOURCE_DIR}/cmake)

# PROJECT_IS_TOP_LEVEL is available since 3.21
get_property(not_top DIRECTORY PROPERTY PARENT_DIRECTORY)
if(not_top AND ONNXRUNTIME_ROOT)
  set(_ONNXRUNTIME_EMBEDDED TRUE)
endif()

if(OCOS_ENABLE_SELECTED_OPLIST)
  # Need to ensure _selectedoplist.cmake file is already generated in folder: ${PROJECT_SOURCE_DIR}/cmake/
  # You could run gen_selectedops.py in folder: tools/ to generate _selectedoplist.cmake
  message(STATUS "Looking for the _selectedoplist.cmake")
  disable_all_operators()
  include(_selectedoplist)
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
    message(WARNING "Enabling C++ exception support as custom ops included in the build require.")
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
  # make sure we're not building with a 3rd party library that requires exceptions
  if (NOT OCOS_ENABLE_CPP_EXCEPTIONS)
      if (OCOS_ENABLE_GPT2_TOKENIZER)
          # gpt2 tokenizer depends on nlohmann_json in onnxruntime, which is old and cannot disable exceptions.
          # could remove this limit when the nlohmann_json is updated in onnxruntime.
          message(FATAL_ERROR "GPT2_TOKENIZER operator needs C++ exceptions support")
      endif()
      if (OCOS_ENABLE_WORDPIECE_TOKENIZER)
          # wordpiece tokenizer depends on nlohmann_json in onnxruntime, which is old and cannot disable exceptions.
          # could remove this limit when the nlohmann_json is updated in onnxruntime.
          message(FATAL_ERROR "WORDPIECE_TOKENIZER operator needs C++ exceptions support")
      endif()
      if (OCOS_ENABLE_BLINGFIRE)
          message(FATAL_ERROR "BLINGFIRE operator needs C++ exceptions support")
      endif()
      if (OCOS_ENABLE_SPM_TOKENIZER)
          message(FATAL_ERROR "SPM_TOKENIZER operator needs C++ exceptions support")
      endif()
      if (OCOS_ENABLE_CV2 OR OCOS_ENABLE_OPENCV_CODECS OR OCOS_ENABLE_VISION)
          message(FATAL_ERROR "The operators depending on OpenCV need C++ exceptions support")
      endif()
  endif()

  add_compile_definitions(OCOS_NO_EXCEPTIONS ORT_NO_EXCEPTIONS)

  if(MSVC)
    string(REPLACE "/EHsc" "/EHs-c-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    add_compile_definitions("_HAS_EXCEPTIONS=0")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables")
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

# set default MSVC warning level to 3 for external dependencies
set_msvc_c_cpp_compiler_warning_level(3)
include(ext_ortlib)
include(gsl)

macro(standardize_output_folder bin_target)
  set_target_properties(${bin_target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endmacro()

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
set(TARGET_SRC_NOEXCEPTION)
file(GLOB TARGET_SRC "operators/*.cc" "operators/*.h" "includes/*.h*")

if(OCOS_ENABLE_TF_STRING)
  set(farmhash_SOURCE_DIR ${PROJECT_SOURCE_DIR}/cmake/externals/farmhash)
  file(GLOB TARGET_SRC_KERNELS "operators/text/*.cc" "operators/text/*.h*")
  file(GLOB TARGET_SRC_HASH "${farmhash_SOURCE_DIR}/src/farmhash.*")
  list(APPEND TARGET_SRC_NOEXCEPTION ${TARGET_SRC_KERNELS} ${TARGET_SRC_HASH})
endif()

if(OCOS_ENABLE_RE2_REGEX)
  file(GLOB TARGET_SRC_RE2_KERNELS "operators/text/re2_strings/*.cc" "operators/text/re2_strings/*.h*")
  list(APPEND TARGET_SRC_NOEXCEPTION ${TARGET_SRC_RE2_KERNELS})
endif()

if(OCOS_ENABLE_MATH)
  if(OCOS_ENABLE_DLIB)
    set(DLIB_ISO_CPP_ONLY ON CACHE INTERNAL "")
    set(DLIB_NO_GUI_SUPPORT ON CACHE INTERNAL "")
    set(DLIB_USE_CUDA OFF CACHE INTERNAL "")
    set(DLIB_USE_LAPACK OFF CACHE INTERNAL "")
    set(DLIB_USE_BLAS OFF CACHE INTERNAL "")
    include(dlib)

    # Ideally, dlib should be included as
    # file(GLOB TARGET_SRC_DLIB "${dlib_SOURCE_DIR}/dlib/all/source.cpp")
    # To avoid the unintentional using some unwanted component, only include
    file(GLOB TARGET_SRC_DLIB "${dlib_SOURCE_DIR}/dlib/test_for_odr_violations.cpp")
    file(GLOB TARGET_SRC_INVERSE "operators/math/dlib/*.cc" "operators/math/dlib/*.h*")
  endif()

  file(GLOB TARGET_SRC_MATH "operators/math/*.cc" "operators/math/*.h*")
  list(APPEND TARGET_SRC ${TARGET_SRC_MATH} ${TARGET_SRC_DLIB} ${TARGET_SRC_INVERSE})
endif()

# enable the opencv dependency if we have ops that require it
if(OCOS_ENABLE_CV2 OR OCOS_ENABLE_VISION)
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
  if(NOT OCOS_ENABLE_OPENCV_CODECS)
    message(FATAL_ERROR "OCOS_ENABLE_VISION requires OCOS_ENABLE_OPENCV_CODECS to be ON")
  endif()

  file(GLOB TARGET_SRC_VISION "operators/vision/*.cc" "operators/vision/*.h*")
  list(APPEND TARGET_SRC ${TARGET_SRC_VISION})
endif()

set(_HAS_TOKENIZER OFF)

if(OCOS_ENABLE_GPT2_TOKENIZER)
  # GPT2
  set(_HAS_TOKENIZER ON)
  file(GLOB tok_TARGET_SRC "operators/tokenizer/gpt*.cc" "operators/tokenizer/unicode*.*" "operators/tokenizer/clip*.cc")
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
  file(GLOB bert_TARGET_SRC "operators/tokenizer/basic_tokenizer.*" "operators/tokenizer/bert_tokenizer.*" "operators/tokenizer/bert_tokenizer_decoder.*")
  list(APPEND TARGET_SRC ${bert_TARGET_SRC})
endif()

if(OCOS_ENABLE_BLINGFIRE)
  # blingfire
  set(_HAS_TOKENIZER ON)
  file(GLOB blingfire_TARGET_SRC "operators/tokenizer/blingfire*.*")
  list(APPEND TARGET_SRC ${blingfire_TARGET_SRC})
endif()

if(OCOS_ENABLE_GPT2_TOKENIZER OR OCOS_ENABLE_WORDPIECE_TOKENIZER)
  if(NOT TARGET nlohmann_json)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    message(STATUS "Fetch json")
    include(json)
  endif()
endif()

if(_HAS_TOKENIZER)
  message(STATUS "Tokenizer needed.")
  file(GLOB tokenizer_TARGET_SRC "operators/tokenizer/tokenizers.*")
  list(APPEND TARGET_SRC ${tokenizer_TARGET_SRC})
endif()

# ### make all compile options.
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
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
  ${PROJECT_SOURCE_DIR}/includes
  ${PROJECT_SOURCE_DIR}/operators)

target_include_directories(ocos_operators PUBLIC
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/includes
  ${PROJECT_SOURCE_DIR}/operators
  ${PROJECT_SOURCE_DIR}/operators/tokenizer)

set(ocos_libraries)
set(OCOS_COMPILE_DEFINITIONS)

if(OCOS_ENABLE_DLIB)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_DLIB)
endif()

if(_HAS_TOKENIZER)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_TOKENIZER)
endif()

if(OCOS_ENABLE_TF_STRING)
  target_include_directories(noexcep_operators PUBLIC
    ${googlere2_SOURCE_DIR}
    ${farmhash_SOURCE_DIR}/src)
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_TF_STRING NOMINMAX FARMHASH_NO_BUILTIN_EXPECT FARMHASH_DEBUG=0)
  target_link_libraries(noexcep_operators PRIVATE re2)
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
endif()

if(OCOS_ENABLE_GPT2_TOKENIZER)
  # GPT2
  list(APPEND OCOS_COMPILE_DEFINITIONS ENABLE_GPT2_TOKENIZER)
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
  target_include_directories(ocos_operators PRIVATE ${json_SOURCE_DIR}/single_include)
  list(APPEND ocos_libraries nlohmann_json::nlohmann_json)
endif()

target_include_directories(ocos_operators PRIVATE ${GSL_INCLUDE_DIR})
list(APPEND ocos_libraries Microsoft.GSL::GSL)

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

file(GLOB shared_TARGET_LIB_SRC "shared/lib/*.cc" "shared/lib/*.h")

if(NOT OCOS_ENABLE_STATIC_LIB AND CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  add_executable(ortcustomops ${shared_TARGET_LIB_SRC})
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
  add_library(ortcustomops STATIC ${shared_TARGET_LIB_SRC})
  if (HAS_SDL)
    target_compile_options(ortcustomops PRIVATE "/sdl")
  endif()
  add_library(onnxruntime_extensions ALIAS ortcustomops)
  standardize_output_folder(ortcustomops)
  set(_BUILD_SHARED_LIBRARY TRUE)
endif()

target_compile_definitions(ortcustomops PUBLIC ${OCOS_COMPILE_DEFINITIONS})
target_include_directories(ortcustomops PUBLIC
  "$<TARGET_PROPERTY:ocos_operators,INTERFACE_INCLUDE_DIRECTORIES>")
target_link_libraries(ortcustomops PUBLIC ocos_operators)

if(_BUILD_SHARED_LIBRARY)
  file(GLOB shared_TARGET_SRC "shared/*.cc" "shared/*.h" "shared/*.def")
  add_library(extensions_shared SHARED ${shared_TARGET_SRC})
  source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${shared_TARGET_SRC})
  standardize_output_folder(extensions_shared)

  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    if(OCOS_ENABLE_SPM_TOKENIZER)
      target_link_libraries(extensions_shared PUBLIC log)
    endif()
  endif()

  if(LINUX OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    set_property(TARGET extensions_shared APPEND_STRING PROPERTY LINK_FLAGS "-Wl,-s -Wl,--version-script -Wl,${PROJECT_SOURCE_DIR}/shared/ortcustomops.ver")
  endif()

  target_include_directories(extensions_shared PUBLIC
    "$<TARGET_PROPERTY:ortcustomops,INTERFACE_INCLUDE_DIRECTORIES>")
  target_link_libraries(extensions_shared PRIVATE ortcustomops)
  set_target_properties(extensions_shared PROPERTIES OUTPUT_NAME "ortextensions")
  if(MSVC AND ocos_target_platform MATCHES "x86|x64")
    target_link_options(extensions_shared PRIVATE "/CETCOMPAT")
  endif()

endif()

if(OCOS_BUILD_PYTHON)
  message(STATUS "Python Build is enabled")
  include(ext_python)
endif()

if(OCOS_BUILD_JAVA)
  message(STATUS "Java Build is enabled")
  include(ext_java)
endif()

if(OCOS_BUILD_APPLE_FRAMEWORK)
  include(ext_apple_framework)
endif()

# clean up the requirements.txt files from 3rd party project folder to suppress the code security false alarms
file(GLOB_RECURSE NO_USE_FILES ${CMAKE_BINARY_DIR}/_deps/*requirements.txt)
message(STATUS "Found the following requirements.txt: ${NO_USE_FILES}")

foreach(nf ${NO_USE_FILES})
  execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${nf})
endforeach()

# test section
if(OCOS_ENABLE_CTEST)
  if (OCOS_ENABLE_SELECTED_OPLIST)
    # currently the tests don't handle operator exclusion cleanly.
    message(FATAL_ERROR "Due to usage of OCOS_ENABLE_SELECTED_OPLIST excluding operators the tests are unable to be built and run")
  endif()

  # Enable CTest
  enable_testing()
  message(STATUS "Fetch CTest")
  include(CTest)

  set(TEST_SRC_DIR ${PROJECT_SOURCE_DIR}/test)
  message(STATUS "Fetch googletest")
  include(googletest)
  file(GLOB static_TEST_SRC "${TEST_SRC_DIR}/static_test/*.cc")
  add_executable(ocos_test ${static_TEST_SRC})
  standardize_output_folder(ocos_test)
  target_link_libraries(ocos_test PRIVATE gtest_main ocos_operators ${ocos_libraries})
  add_test(NAME ocos_test COMMAND $<TARGET_FILE:ocos_test>)

  SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
  find_library(ONNXRUNTIME onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}")

  if(ONNXRUNTIME-NOTFOUND)
    message(WARNING "The prebuilt onnxruntime libraries directory cannot found (via ONNXRUNTIME_LIB_DIR), the extensions_test will be skipped.")
  else()
    set(LINUX_CC_FLAGS "")

    # needs to link with stdc++fs in Linux
    if(UNIX AND NOT APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND LINUX_CC_FLAGS stdc++fs -pthread)
    endif()

    file(GLOB shared_TEST_SRC "${TEST_SRC_DIR}/shared_test/*.cc")
    add_executable(extensions_test ${shared_TEST_SRC})
    standardize_output_folder(extensions_test)
    target_include_directories(extensions_test PRIVATE ${spm_INCLUDE_DIRS}
      "$<TARGET_PROPERTY:extensions_shared,INTERFACE_INCLUDE_DIRECTORIES>")

    if(ONNXRUNTIME_LIB_DIR)
      target_link_directories(extensions_test PRIVATE ${ONNXRUNTIME_LIB_DIR})
    endif()

    target_link_libraries(extensions_test PRIVATE ocos_operators extensions_shared onnxruntime gtest_main gmock_main
                          ${ocos_libraries} ${LINUX_CC_FLAGS})

    # Copy ONNXRuntime DLLs into bin folder for testing on Windows platform
    if(WIN32)
      file(TO_CMAKE_PATH "${ONNXRUNTIME_LIB_DIR}/*" ONNXRUNTIME_LIB_FILEPATTERN)
      file(GLOB ONNXRUNTIME_LIB_FILES CONFIGURE_DEPENDS "${ONNXRUNTIME_LIB_FILEPATTERN}")
      add_custom_command(
        TARGET extensions_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_LIB_FILES} $<TARGET_FILE_DIR:extensions_test>)
    endif()

    set(TEST_DATA_SRC ${TEST_SRC_DIR}/data)
    set(TEST_DATA_DES ${onnxruntime_extensions_BINARY_DIR}/data)

    # Copy test data from source to destination.
    add_custom_command(
      TARGET extensions_test POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${TEST_DATA_SRC}
      ${TEST_DATA_DES})
    add_test(NAME extensions_test COMMAND $<TARGET_FILE:extensions_test>)
  endif()
endif()
