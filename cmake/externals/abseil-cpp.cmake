# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(FetchContent)

# Pass to build
set(ABSL_PROPAGATE_CXX_STD 1)
set(BUILD_TESTING 0)
set(ABSL_BUILD_TESTING OFF)
set(ABSL_BUILD_TEST_HELPERS OFF)
set(ABSL_USE_EXTERNAL_GOOGLETEST ON)
if(Patch_FOUND AND WIN32)
  set(ABSL_PATCH_COMMAND ${Patch_EXECUTABLE} --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/cmake/externals/absl.patch)
else()
  set(ABSL_PATCH_COMMAND "")
endif()
set(ABSL_ENABLE_INSTALL ON)
fetchcontent_declare(
    abseil_cpp
    URL https://github.com/abseil/abseil-cpp/archive/refs/tags/20240722.0.zip
    URL_HASH SHA1=36ee53eb1466fb6e593fc5c286680de31f8a494a
    EXCLUDE_FROM_ALL
    PATCH_COMMAND ${ABSL_PATCH_COMMAND}
    FIND_PACKAGE_ARGS 20240722 NAMES absl
)

Fetchcontent_makeavailable(abseil_cpp)