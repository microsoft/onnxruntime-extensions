FetchContent_Declare(
  googlere2
  GIT_REPOSITORY https://github.com/google/re2.git
  GIT_TAG        2020-11-01
)

set(RE2_BUILD_TESTING OFF)
# google/re2 is using a too general name for this option
set(BUILD_SHARED_LIBS ON)
FetchContent_MakeAvailable(googlere2)
