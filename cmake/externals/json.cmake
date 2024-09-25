FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.10.5
  SOURCE_SUBDIR not_set
  )

add_compile_definitions(JSON_HAS_CPP_17=1)
FetchContent_MakeAvailable(nlohmann_json)
