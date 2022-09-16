FetchContent_Declare(
  spm
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.1.96
)
# spm is abbr. of sentencepiece to meet the MAX_PATH compiling requirement on Windows
FetchContent_GetProperties(spm)

if(NOT spm_POPULATED)
  FetchContent_Populate(spm)
  # need to patch sentencepiece to use protobuf provided by ort. This is ifdef onnxruntime_BUILD_WEBASSEMBLY
  # for now but there is no real reason to not use it for all builds.
  # 'git apply' should be in FetchContent_Declare() but that creates issues in sucessive builds so for now it is using execute_process().
  if(onnxruntime_BUILD_WEBASSEMBLY)
    set(SPM_PATCH "${PROJECT_SOURCE_DIR}/cmake/externals/sentencepieceproject.patch")
    message("-- sentencepiece: patching with ${SPM_PATCH}")
    execute_process(COMMAND git apply --ignore-space-change --ignore-whitespace ${SPM_PATCH} WORKING_DIRECTORY ${spm_SOURCE_DIR})
  endif()
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
