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
    GIT_TAG v25.7
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS NAMES Protobuf protobuf
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
  if(TARGET libprotobuf-lite)
    set_target_properties(libprotobuf-lite PROPERTIES
    FOLDER externals/google)
  endif()
endif()

# To avoid creating complicated logic to build protoc, especially for mobile platforms, we use the pre-generated pb files
# Uses the following command line in _deps/spm-src folder to generate the PB patch file if protobuf version is updated
# git diff -- src/builtin_pb/* | out-file -Encoding utf8 <REPO-ROOT>\cmake\externals\sentencepieceproject_pb.patch
# PB files was seperated as another patch file to avoid the patch file too large to be reviewed.

set(spm_patch_command ${Patch_EXECUTABLE} -p1 <  "${PROJECT_SOURCE_DIR}/cmake/externals/sentencepiece.patch")

if (NOT DEFINED CMAKE_INSTALL_INCDIR)
  set(CMAKE_INSTALL_INCDIR include)
endif()

FetchContent_Declare(
  spm
  URL https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz
  URL_HASH SHA512=b4214f5bfbe2a0757794c792e87e7c53fda7e65b2511b37fc757f280bf9287ba59b5d630801e17de6058f8292a3c6433211917324cb3446a212a51735402e614
  EXCLUDE_FROM_ALL
  PATCH_COMMAND ${spm_patch_command}
  FIND_PACKAGE_ARGS NAMES sentencepiece
)

set(SPM_ABSL_PROVIDER "package" CACHE STRING "Protobuf provider" FORCE)
set(SPM_PROTOBUF_PROVIDER "package" CACHE STRING "Protobuf provider" FORCE)


FetchContent_MakeAvailable(spm)
if(sentencepiece_SOURCE_DIR)
  message(STATUS "XXXXXXXXXXXXXXXX sentencepiece source dir: ${sentencepiece_SOURCE_DIR}")
  include_directories(${sentencepiece_SOURCE_DIR}/src)
endif()

