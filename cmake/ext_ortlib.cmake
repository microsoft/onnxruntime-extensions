if(_ONNXRUNTIME_EMBEDDED)
  set(ONNXRUNTIME_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../include/onnxruntime/core/session)
  set(ONNXRUNTIME_LIB_DIR "")
elseif(ONNXRUNTIME_PKG_DIR)
  set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_PKG_DIR}/include)
  set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_PKG_DIR}/lib)
else()
  message(STATUS "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
  message(STATUS "CMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}")

  # default to 1.11.1 if not specified
  set(ONNXRUNTIME_VER "1.11.1" CACHE STRING "ONNX Runtime version")

  if(APPLE)
    set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-osx-universal2-${ONNXRUNTIME_VER}.tgz")
  elseif(WIN32)
    set(ONNXRUNTIME_BINARY_PLATFORM "x64")

    # override if generator platform is set
    if (CMAKE_GENERATOR_PLATFORM)
      if (CMAKE_GENERATOR_PLATFORM STREQUAL "Win32")
        set(ONNXRUNTIME_BINARY_PLATFORM "x86")
      elseif (CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64" OR CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64EC")
        set(ONNXRUNTIME_BINARY_PLATFORM "arm64")
      elseif (CMAKE_GENERATOR_PLATFORM STREQUAL "ARM")
        set(ONNXRUNTIME_BINARY_PLATFORM "arm")
      endif()
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
      # or if building on arm64 machine
      set(ONNXRUNTIME_BINARY_PLATFORM "arm64")
    endif()

    set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-win-${ONNXRUNTIME_BINARY_PLATFORM}-${ONNXRUNTIME_VER}.zip")
  else()
    # Linux or other, using Linux package to retrieve the headers
    set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-linux-x64-${ONNXRUNTIME_VER}.tgz")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VER}.tgz")
    endif()
  endif()

  message(STATUS "ONNX Runtime URL suffix: ${ONNXRUNTIME_URL}")
  FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/${ONNXRUNTIME_URL}
  )

  FetchContent_makeAvailable(onnxruntime)
  set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
  set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
endif()

if(NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR})
  message(FATAL_ERROR "ONNX Runtime headers not found at ${ONNXRUNTIME_INCLUDE_DIR}")
endif()
