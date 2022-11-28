set(PNG_SHARED      OFF CACHE INTERNAL "")
set(PNG_TESTS       OFF CACHE INTERNAL "")
set(PNG_EXECUTABLES OFF CACHE INTERNAL "")
set(PNG_BUILD_ZLIB  ON  CACHE INTERNAL "")

FetchContent_Declare(
  libpng
  GIT_REPOSITORY https://github.com/glennrp/libpng.git
  GIT_TAG        v1.6.39
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(libpng)
