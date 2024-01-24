# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

project(tfmtok_tokenizers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Gather source files
file(GLOB TOK_SOURCES *.cc)
file(GLOB UTILS_SOURCES utils/*.cc)
file(GLOB CAPI_SOURCES c_api/*.cc c_api/*.def)

# Create static library
add_library(tfmtok STATIC ${TOK_SOURCES} ${UTILS_SOURCES})
target_include_directories(tfmtok PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${simdjson_SOURCE_DIR}/singleheader
  ${GSL_SOURCE_DIR}/include)

add_library(tfmtok_c SHARED  ${CAPI_SOURCES})
add_dependencies(tfmtok_c tfmtok)
target_link_libraries(tfmtok_c PUBLIC tfmtok)
