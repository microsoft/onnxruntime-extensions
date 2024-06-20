FetchContent_Declare(
  pybind11
  URL       https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip
  URL_HASH  SHA1=8482f57ed55c7b100672815a311d5450858723fb
)

FetchContent_GetProperties(pybind11)
# Check if population has already been performed
string(TOLOWER "pybind11" lcName)
if(NOT ${lcName}_POPULATED)
  # Fetch the content using previously declared details
  FetchContent_Populate(pybind11)
endif()

set(pybind11_INCLUDE_DIRS ${pybind11_SOURCE_DIR}/include)
