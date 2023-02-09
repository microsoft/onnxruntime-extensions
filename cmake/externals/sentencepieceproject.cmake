if(NOT _ONNXRUNTIME_EMBEDDED)
  # TOOD: migrate to external abseil library
  # include(abseil-cpp)

  find_package(Patch)
  if(Patch_FOUND)
    message("Patch found: ${Patch_EXECUTABLE}")
  endif()

  FetchContent_Declare(
    Protobuf
    URL https://github.com/protocolbuffers/protobuf/archive/refs/tags/v3.18.3.zip
    URL_HASH SHA1=b95bf7e9de9c2249b6c1f2ca556ace49999e90bd
    SOURCE_SUBDIR cmake
    FIND_PACKAGE_ARGS 3.18.0 NAMES Protobuf
    PATCH_COMMAND ${Patch_EXECUTABLE} --binary -l -s -p1 -i ${PROJECT_SOURCE_DIR}/cmake/externals/protobuf.patch
  )
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
  set(protobuf_WITH_ZLIB OFF CACHE BOOL "Use zlib" FORCE)

  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE "" FORCE)
  endif()
  
  set(protobuf_DISABLE_RTTI ON CACHE BOOL "Disable RTTI" FORCE)
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

set(spm_patch "${PROJECT_SOURCE_DIR}/cmake/externals/sentencepieceproject.protobuf.patch")

set(spm_patch_command)
if(spm_patch)
  set(spm_patch_command ${Patch_EXECUTABLE} --binary -s -l -p1 -i ${spm_patch})
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
