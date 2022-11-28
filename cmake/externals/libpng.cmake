FetchContent_Declare(
  libpng
  GIT_REPOSITORY https://github.com/glennrp/libpng.git
  GIT_TAG        v1.6.39
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(libpng)
