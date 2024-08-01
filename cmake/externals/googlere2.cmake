FetchContent_Declare(
  googlere2
  GIT_REPOSITORY https://github.com/google/re2.git
  GIT_TAG        2021-06-01
  EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(googlere2)
set_target_properties(re2
  PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      FOLDER externals/google)
