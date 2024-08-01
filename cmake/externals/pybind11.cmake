FetchContent_Declare(
  pybind11
  URL       https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip
  URL_HASH  SHA1=8482f57ed55c7b100672815a311d5450858723fb
  SOURCE_SUBDIR not_set
)

FetchContent_MakeAvailable(pybind11)

set(pybind11_INCLUDE_DIRS ${pybind11_SOURCE_DIR}/include)
