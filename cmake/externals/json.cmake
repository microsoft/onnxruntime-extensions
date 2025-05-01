if (OCOS_USE_BUILTIN_DEPS)
    set(nlohmann_json_SOURCE_DIR "${CMAKE_SOURCE_DIR}/cmake/externals/nlohmann_json")
else()
  FetchContent_Declare(nlohmann_json
      URL       https://codeload.github.com/nlohmann/json/zip/refs/tags/v3.11.3
      URL_HASH  SHA1=5e88795165cc8590138d1f47ce94ee567b85b4d6
      SOURCE_SUBDIR not_set
  )

  FetchContent_MakeAvailable(nlohmann_json)
endif()
add_compile_definitions(JSON_HAS_CPP_17=1)
