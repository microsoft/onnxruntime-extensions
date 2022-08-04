FetchContent_Declare(
  spm
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.1.96
)
# spm is abbr. of sentencepiece to meet the MAX_PATH compiling requirement on Windows
FetchContent_GetProperties(spm)

if(NOT spm_POPULATED)
  FetchContent_Populate(spm)
  add_subdirectory(${spm_SOURCE_DIR} ${spm_BINARY_DIR} EXCLUDE_FROM_ALL)
  set_target_properties(sentencepiece-static PROPERTIES
    FOLDER externals/google/sentencepiece)
endif()

set(spm_INCLUDE_DIRS
    ${spm_SOURCE_DIR}/third_party/protobuf-lite
    ${spm_SOURCE_DIR}/src/builtin_pb
    ${spm_SOURCE_DIR}/third_party
    ${spm_SOURCE_DIR}/src
    )
