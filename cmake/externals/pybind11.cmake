FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.6.0
)

FetchContent_GetProperties(pybind11)
# Check if population has already been performed
string(TOLOWER "pybind11" lcName)
if(NOT ${lcName}_POPULATED)
  # Fetch the content using previously declared details
  FetchContent_Populate(pybind11)
endif()

set(pybind11_INCLUDE_DIRS ${pybind11_SOURCE_DIR}/include)
