FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.13.0
  GIT_SHALLOW TRUE
  GIT_SUBMODULES_RECURSE TRUE
)

set(NB_TEST OFF CACHE BOOL "" FORCE)
set(NB_USE_SUBMODULE_DEPS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nanobind)
