# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(CUDAToolkit)
enable_language(CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_STANDARD 17)

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
