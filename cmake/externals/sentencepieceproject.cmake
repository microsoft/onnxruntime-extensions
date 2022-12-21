# spm is abbreviation of sentencepiece to meet the path length limits on Windows

set(spm_patches)
if(_ONNXRUNTIME_EMBEDDED)
  # use protobuf provided by ORT
  list(APPEND spm_patches "${PROJECT_SOURCE_DIR}/cmake/externals/sentencepieceproject.protobuf.patch")
endif()
if(IOS)
  # fix for iOS build
  list(APPEND spm_patches "${PROJECT_SOURCE_DIR}/cmake/externals/sentencepieceproject.ios.patch")
endif()

set(spm_patch_command)
if(spm_patches)
  set(spm_patch_command git checkout . && git apply --ignore-space-change --ignore-whitespace ${spm_patches})
endif()

FetchContent_Declare(
  spm
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.1.96
  PATCH_COMMAND ${spm_patch_command}
)
FetchContent_GetProperties(spm)

if(NOT spm_POPULATED)
  FetchContent_Populate(spm)
  add_subdirectory(${spm_SOURCE_DIR} ${spm_BINARY_DIR} EXCLUDE_FROM_ALL)
  set_target_properties(sentencepiece-static PROPERTIES
    FOLDER externals/google/sentencepiece)
endif()

if(_ONNXRUNTIME_EMBEDDED)
  set(SPM_USE_BUILTIN_PROTOBUF OFF)
  set(spm_INCLUDE_DIRS
    ${REPO_ROOT}/cmake/external/protobuf/src
    ${spm_SOURCE_DIR}/src/builtin_pb
    ${spm_SOURCE_DIR}/third_party
    ${spm_SOURCE_DIR}/src
)
else()
  set(spm_INCLUDE_DIRS
      ${spm_SOURCE_DIR}/third_party/protobuf-lite
      ${spm_SOURCE_DIR}/src/builtin_pb
      ${spm_SOURCE_DIR}/third_party
      ${spm_SOURCE_DIR}/src
      )
endif()
