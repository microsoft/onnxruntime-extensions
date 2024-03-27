# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(CUDAToolkit)
enable_language(CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_STANDARD 17)


if(NOT CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL "aarch64-linux-gnu")
    # Support for Jetson/Tegra ARM devices
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_53,code=sm_53") # TX1, Nano
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_62,code=sm_62") # TX2
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_72,code=sm_72") # AGX Xavier,
                                                                   # NX Xavier
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
      set(CMAKE_CUDA_FLAGS
          "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_87,code=sm_87") # AGX Orin,
                                                                     # NX Orin
    endif()
  else()
    # the following compute capabilities are removed in CUDA 11 Toolkit
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
      set(CMAKE_CUDA_FLAGS
          "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_30,code=sm_30") # K series
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12)
      # 37, 50 still work in CUDA 11 but are marked deprecated and will be
      # removed in future CUDA version.
      set(CMAKE_CUDA_FLAGS
          "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_37,code=sm_37") # K80
      set(CMAKE_CUDA_FLAGS
          "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50") # M series
    endif()
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_52,code=sm_52") # M60
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60") # P series
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # V series
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75") # T series
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
      set(CMAKE_CUDA_FLAGS
          "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80") # A series
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
      set(CMAKE_CUDA_FLAGS
          "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90,code=sm_90") # H series
    endif()
  endif()
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
