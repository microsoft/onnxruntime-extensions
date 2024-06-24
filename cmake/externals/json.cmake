FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.10.5)

set(JSON_BuildTests OFF CACHE INTERNAL "")

FetchContent_GetProperties(nlohmann_json)
if(NOT nlohmann_json_POPULATED)
  FetchContent_Populate(nlohmann_json)
endif()

add_compile_definitions(JSON_HAS_CPP_17=1)
