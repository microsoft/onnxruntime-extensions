include(FetchContent)

FetchContent_Declare(
  googlere2
  GIT_REPOSITORY https://github.com/google/re2.git
  GIT_TAG        2020-11-01
)

set(BUILD_SHARED_LIBS ON) # google/re2 using a general name.
FetchContent_MakeAvailable(googlere2)
