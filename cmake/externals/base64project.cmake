FetchContent_Declare(
  base64project
  GIT_REPOSITORY https://github.com/ReneNyffenegger/cpp-base64.git
)

FetchContent_GetProperties(base64project)

if(NOT base64project_POPULATED)
  FetchContent_Populate(base64project)
endif()

file(GLOB base64_TARGET_SRC
    "${base64project_SOURCE_DIR}/base64.cpp"
    "${base64project_SOURCE_DIR}/base64.h")
