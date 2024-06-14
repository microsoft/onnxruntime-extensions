# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(CUDAToolkit)
enable_language(CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_STANDARD 17)
cmake_dependent_option(OCOS_USE_FLASH_ATTENTION "Build flash attention kernel for scaled dot product attention" ON "NOT WIN32" OFF)
option(OCOS_USE_MEMORY_EFFICIENT_ATTENTION "Build memory efficient attention kernel for scaled dot product attention" ON)
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.6)
  message(STATUS "Turn off flash attention and memory efficient attention since CUDA compiler version < 11.6")
  set(OCOS_USE_FLASH_ATTENTION OFF)
  set(OCOS_USE_MEMORY_EFFICIENT_ATTENTION OFF)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror default-stream-launch")
endif()

if(NOT WIN32)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)
endif()

# Options passed to cudafe
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=bad_friend_decl\"")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=unsigned_compare_with_zero\"")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=expr_has_no_effect\"")

add_compile_definitions(USE_CUDA)

set(OCOS_USE_MEMORY_EFFICIENT_ATTENTION OFF) # turn off for the build time. Turn them on when these 2 libs are really in use
set(OCOS_USE_FLASH_ATTENTION OFF)
if (OCOS_USE_FLASH_ATTENTION)
  message(STATUS "Enable flash attention")
  add_compile_definitions(OCOS_USE_FLASH_ATTENTION)
endif()
if (OCOS_USE_MEMORY_EFFICIENT_ATTENTION)
  message(STATUS "Enable memory efficient attention")
  add_compile_definitions(OCOS_USE_MEMORY_EFFICIENT_ATTENTION)
endif()
