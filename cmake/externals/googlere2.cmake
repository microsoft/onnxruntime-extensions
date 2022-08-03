FetchContent_Declare(
  googlere2
  GIT_REPOSITORY https://github.com/google/re2.git
  GIT_TAG        2020-11-01
)

FetchContent_GetProperties(googlere2)
string(TOLOWER "googlere2" lcName)
if(NOT ${lcName}_POPULATED)
  FetchContent_Populate(googlere2)
  add_subdirectory(${googlere2_SOURCE_DIR} ${googlere2_BINARY_DIR} EXCLUDE_FROM_ALL)
  set_target_properties(re2
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        FOLDER externals/google/re2)
endif()
