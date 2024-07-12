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
    GIT_TAG v3.20.3
    EXCLUDE_FROM_ALL
    PATCH_COMMAND git checkout . && git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/cmake/externals/protobuf_cmake.patch
    SOURCE_SUBDIR cmake
  )
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests")
  set(protobuf_WITH_ZLIB OFF CACHE BOOL "Use zlib")

  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "")
  endif()
  set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "")
  if("${CMAKE_MSVC_RUNTIME_LIBRARY}" STREQUAL "" OR "${CMAKE_MSVC_RUNTIME_LIBRARY}" MATCHES "DLL$")
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "")
  else()
    set(protobuf_MSVC_STATIC_RUNTIME ON CACHE BOOL "")
  endif()
  set(protobuf_DISABLE_RTTI ON CACHE BOOL "Disable RTTI")

  FetchContent_MakeAvailable(protobuf)
  set_target_properties(libprotobuf-lite PROPERTIES
    FOLDER externals/google)
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
  EXCLUDE_FROM_ALL
  PATCH_COMMAND ${spm_patch_command}
)

set(SPM_USE_EXTERNAL_ABSL OFF CACHE BOOL "Use external absl" FORCE)
set(SPM_USE_BUILTIN_PROTOBUF OFF CACHE BOOL "Use built-in protobuf" FORCE)

if(NOT protobuf_SOURCE_DIR)
  message(FATAL_ERROR "Cannot find the protobuf library in ORT")
endif()

FetchContent_MakeAvailable(spm)
target_link_libraries(sentencepiece-static PUBLIC protobuf::libprotobuf-lite)
set_target_properties(sentencepiece-static PROPERTIES
  FOLDER externals/google)

set(spm_INCLUDE_DIRS
  ${protobuf_SOURCE_DIR}/src
  ${spm_SOURCE_DIR}/src/builtin_pb
  ${spm_SOURCE_DIR}/src )
