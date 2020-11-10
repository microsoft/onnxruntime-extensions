include(FetchContent)

FetchContent_Declare(
  googlere2
  GIT_REPOSITORY https://github.com/google/re2.git
  GIT_TAG        2020-11-01
)

set(googlere2_BUILD_TESTS OFF)
set(BUILD_SHARED_LIBS ON)
FetchContent_MakeAvailable(googlere2)
