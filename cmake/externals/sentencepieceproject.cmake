# spm is abbreviation of sentencepiece to meet the path length limits on Windows
if(NOT _ONNXRUNTIME_EMBEDDED)
  # If extensions wasn't built in ORT, we create fetchcontent the same 3rd party library as ORT
  # So extensions is always consistent on the 3rd party libraries whether its build in ORT or not

  # TOOD: migrate to external abseil library
  # include(abseil-cpp)
  message(STATUS "Fetch protobuf")
  FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG v3.20.2
    PATCH_COMMAND git checkout . && git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/cmake/externals/protobuf_cmake.patch
  )
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests")
  set(protobuf_WITH_ZLIB OFF CACHE BOOL "Use zlib")

  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "")
  endif()
  set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "")
  set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "")
  set(protobuf_DISABLE_RTTI ON CACHE BOOL "Disable RTTI")
  FetchContent_GetProperties(protobuf)
  if(NOT protobuf_POPULATED)
    FetchContent_Populate(protobuf)
    add_subdirectory(${protobuf_SOURCE_DIR}/cmake ${protobuf_BINARY_DIR} EXCLUDE_FROM_ALL)  
  endif()

  FetchContent_MakeAvailable(protobuf)
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

# To avoid creating complicated logic to build protoc, especially for mobile platforms, we use the pre-generated pb files
# Uses the following command line in _deps/spm-src folder to generate the PB patch file if protobuf version is updated
# git diff -- src/builtin_pb/* | out-file -Encoding utf8 <REPO-ROOT>\cmake\externals\sentencepieceproject_pb.patch
# PB files was seperated as another patch file to avoid the patch file too large to be reviewed.
set(spm_patches
  "${PROJECT_SOURCE_DIR}/cmake/externals/sentencepieceproject_cmake.patch"
  "${PROJECT_SOURCE_DIR}/cmake/externals/sentencepieceproject_pb.patch")

set(spm_patch_command git checkout . && git apply --ignore-space-change --ignore-whitespace ${spm_patches})

if (NOT DEFINED CMAKE_INSTALL_INCDIR)
  set(CMAKE_INSTALL_INCDIR include)
endif()

FetchContent_Declare(
  spm
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.1.96
  PATCH_COMMAND ${spm_patch_command}
)
FetchContent_GetProperties(spm)

set(SPM_USE_EXTERNAL_ABSL OFF CACHE BOOL "Use external absl" FORCE)
set(SPM_USE_BUILTIN_PROTOBUF OFF CACHE BOOL "Use built-in protobuf" FORCE)

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
