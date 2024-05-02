if (USE_FLASH_ATTENTION OR USE_MEMORY_EFFICIENT_ATTENTION)
  include(FetchContent)
  FetchContent_Declare(
          cutlass
          GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
          GIT_TAG v3.1.0
  )
  
  FetchContent_GetProperties(cutlass)
  if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
  endif()
endif()