FetchContent_Declare(
  sentencepieceproject
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
)

FetchContent_GetProperties(sentencepieceproject)

if(NOT sentencepieceproject_POPULATED)
  FetchContent_Populate(sentencepieceproject)
  add_subdirectory(
    ${sentencepieceproject_SOURCE_DIR}
    ${sentencepieceproject_BINARY_DIR}
    EXCLUDE_FROM_ALL)
endif()

set(sentencepieceproject_INCLUDE_DIRS
    ${sentencepieceproject_SOURCE_DIR}/third_party/protobuf-lite
    ${sentencepieceproject_SOURCE_DIR}/src/builtin_pb
    ${sentencepieceproject_SOURCE_DIR}/third_party    
    ${sentencepieceproject_SOURCE_DIR}/src
    ${sentencepieceproject_SOURCE_DIR}
    ${sentencepieceproject_BINARY_DIR})
