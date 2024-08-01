FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG v3.1.0
  EXCLUDE_FROM_ALL)
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "")
FetchContent_MakeAvailable(cutlass)
