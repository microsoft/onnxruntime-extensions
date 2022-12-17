FetchContent_Declare(
  spm
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.1.96
)
# spm is abbr. of sentencepiece to meet the MAX_PATH compiling requirement on Windows
FetchContent_GetProperties(spm)

if(NOT spm_POPULATED)
  FetchContent_Populate(spm)
  # 'git apply' should be in FetchContent_Declare() but that creates issues in sucessive builds so for now it is using execute_process().
  foreach(SPM_PATCH IN ITEMS
          # use protobuf provided by ORT
          "${PROJECT_SOURCE_DIR}/cmake/externals/sentencepieceproject.protobuf.patch"
          # fix for iOS build
          "${PROJECT_SOURCE_DIR}/cmake/externals/sentencepieceproject.ios.patch"
          )
    message("-- sentencepiece: patching with ${SPM_PATCH}")
    execute_process(COMMAND git apply --ignore-space-change --ignore-whitespace ${SPM_PATCH} WORKING_DIRECTORY ${spm_SOURCE_DIR})
  endforeach()
  add_subdirectory(${spm_SOURCE_DIR} ${spm_BINARY_DIR} EXCLUDE_FROM_ALL)
  set_target_properties(sentencepiece-static PROPERTIES
    FOLDER externals/google/sentencepiece)
endif()

if (onnxruntime_BUILD_WEBASSEMBLY)
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
