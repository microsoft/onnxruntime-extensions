FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
)

FetchContent_MakeAvailable(sentencepiece)

if(NOT sentencepiece_POPULATED)
  FetchContent_Populate(sentencepiece)
  add_subdirectory(${sentencepiece_SOURCE_DIR} ${sentencepiece_BINARY_DIR})
  set_target_properties(sentencepiece PROPERTIES SPM_ENABLE_SHARED OFF)
endif()

set(sentencepiece_INCLUDE_DIRS
    ${sentencepiece_SOURCE_DIR}/third_party/protobuf-lite
    ${sentencepiece_SOURCE_DIR}/src/builtin_pb
    ${sentencepiece_SOURCE_DIR}/third_party    
    ${sentencepiece_SOURCE_DIR}/src
    )
