if(_ONNXRUNTIME_EMBEDDED)
  set(ONNXRUNTIME_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include/onnxruntime/core/session)
  set(ONNXRUNTIME_LIB_DIR "")
else()
  set(ONNXRUNTIME_VER "1.10.0" CACHE STRING "ONNX Runtime version")

  if(CMAKE_HOST_APPLE)
    set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-osx-universal2-${ONNXRUNTIME_VER}.tgz")
  elseif(CMAKE_HOST_WIN32)
    set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-win-x64-${ONNXRUNTIME_VER}.zip")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
      set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-win-arm64-${ONNXRUNTIME_VER}.zip")
    endif()
  else()
    # Linux or other, using Linux package to retrieve the headers
    set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-linux-x64-${ONNXRUNTIME_VER}.tgz")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(ONNXRUNTIME_URL "v${ONNXRUNTIME_VER}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VER}.tgz")
    endif()
  endif()

  # Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
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
