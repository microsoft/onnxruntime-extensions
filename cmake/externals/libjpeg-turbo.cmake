include(FetchContent)
FetchContent_Declare(
  libjpeg-turbo
  GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo.git
  GIT_TAG        2.1.4
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(libjpeg-turbo)
