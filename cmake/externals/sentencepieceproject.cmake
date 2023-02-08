if(NOT _ONNXRUNTIME_EMBEDDED)
  include(abseil-cpp)

  FetchContent_Declare(
    Protobuf
    URL https://github.com/protocolbuffers/protobuf/archive/refs/tags/v3.18.3.zip
    URL_HASH SHA1=b95bf7e9de9c2249b6c1f2ca556ace49999e90bd
    SOURCE_SUBDIR cmake
    FIND_PACKAGE_ARGS 3.18.0 NAMES Protobuf
    PATCH_COMMAND git checkout . && git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/cmake/externals/protobuf.patch
  )
  set(protobuf_BUILD_TESTS OFF)
  set(protobuf_WITH_ZLIB OFF)

  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(protobuf_BUILD_PROTOC_BINARIES OFF)
  endif()
  
  set(protobuf_DISABLE_RTTI ON)
  FetchContent_MakeAvailable(Protobuf)
  set_target_properties(libprotobuf PROPERTIES
    FOLDER externals/google/protobuf)
  set_target_properties(libprotobuf-lite PROPERTIES
    FOLDER externals/google/protobuf)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    set_target_properties(libprotoc PROPERTIES
      FOLDER externals/google/protobuf)
    set_target_properties(protoc PROPERTIES
      FOLDER externals/google/protobuf)
  endif()
endif()

set(spm_patches)
# use protobuf provided by ORT
list(APPEND spm_patches "${PROJECT_SOURCE_DIR}/cmake/externals/sentencepieceproject.protobuf.patch")
if(IOS)
  # fix for iOS build
  list(APPEND spm_patches "${PROJECT_SOURCE_DIR}/cmake/externals/sentencepieceproject.ios.patch")
endif()

set(spm_patch_command)
if(spm_patches)
  set(spm_patch_command git checkout . && git apply --ignore-space-change --ignore-whitespace ${spm_patches})
endif()

if (NOT DEFINED CMAKE_INSTALL_INCDIR)
  set(CMAKE_INSTALL_INCDIR include)
endif()

# spm is abbr. of sentencepiece to meet the MAX_PATH compiling requirement on Windows
FetchContent_Declare(
  spm
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.1.96
  PATCH_COMMAND ${spm_patch_command}
)
FetchContent_GetProperties(spm)

set(SPM_USE_EXTERNAL_ABSL OFF)
set(SPM_USE_BUILTIN_PROTOBUF OFF)

if(NOT protobuf_SOURCE_DIR)
  message(FATAL_ERROR "Cannot find the protobuf library in ORT")
endif()

if(NOT spm_POPULATED)
  FetchContent_Populate(spm)
  add_subdirectory(${spm_SOURCE_DIR} ${spm_BINARY_DIR} EXCLUDE_FROM_ALL)
  target_link_libraries(sentencepiece-static PUBLIC protobuf::libprotobuf-lite)
  set_target_properties(sentencepiece-static PROPERTIES
    FOLDER externals/google/sentencepiece)
endif()

set(spm_INCLUDE_DIRS
  ${protobuf_SOURCE_DIR}/src
  ${spm_SOURCE_DIR}/src/builtin_pb
  ${spm_SOURCE_DIR}/src )
