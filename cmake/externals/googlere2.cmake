FetchContent_Declare(
  googlere2
  GIT_REPOSITORY https://github.com/google/re2.git
  GIT_TAG        2021-06-01
  EXCLUDE_FROM_ALL
  FIND_PACKAGE_ARGS NAMES re2
)

FetchContent_MakeAvailable(googlere2)
if(TARGET re2)
set_target_properties(re2
  PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      FOLDER externals/google)
endif()
