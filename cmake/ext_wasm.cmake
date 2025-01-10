# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(OCOS_ENABLE_STATIC_LIB)
  add_library(ortcustomops STATIC ${_TARGET_LIB_SRC})
  if (HAS_SDL)
    target_compile_options(ortcustomops PRIVATE "/sdl")
  endif()
  add_library(onnxruntime_extensions ALIAS ortcustomops)
  standardize_output_folder(ortcustomops)
else()
  add_executable(ortcustomops ${_TARGET_LIB_SRC})
  set_target_properties(ortcustomops PROPERTIES LINK_FLAGS "                                  \
                      -s WASM=1                                                               \
                      -s NO_EXIT_RUNTIME=0                                                    \
                      -s ALLOW_MEMORY_GROWTH=1                                                \
                      -s SAFE_HEAP=0                                                          \
                      -s MODULARIZE=1                                                         \
                      -s SAFE_HEAP_LOG=0                                                      \
                      -s STACK_OVERFLOW_CHECK=0                                               \
                      -s EXPORT_ALL=0                                                         \
                      -s VERBOSE=0                                                            \
                      --no-entry")

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_property(TARGET ortcustomops APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=1 -s DEMANGLE_SUPPORT=1")
  else()
    set_property(TARGET ortcustomops APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=0 -s DEMANGLE_SUPPORT=0")
  endif()
endif()
