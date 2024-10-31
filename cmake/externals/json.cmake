FetchContent_Declare(nlohmann_json
  URL       https://codeload.github.com/nlohmann/json/zip/refs/tags/v3.10.5
  URL_HASH  SHA1=f257f8dc27c5b8c085dc887b40cddd18ae1f725c
  SOURCE_SUBDIR not_set
  )

add_compile_definitions(JSON_HAS_CPP_17=1)
FetchContent_MakeAvailable(nlohmann_json)
