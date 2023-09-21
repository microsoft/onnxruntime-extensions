if(_ONNXRUNTIME_EMBEDDED)
  set(ONNXRUNTIME_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../include/onnxruntime/core/session)
  set(ONNXRUNTIME_LIB_DIR "")
elseif(ONNXRUNTIME_PKG_DIR)
  set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_PKG_DIR}/include)
  set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_PKG_DIR}/lib)
elseif(OCOS_ONNXRUNTIME_PKG_URI)
  if (NOT OCOS_ONNXRUNTIME_VERSION)
    message(FATAL_ERROR "OCOS_ONNXRUNTIME_PKG_URI is set but OCOS_ONNXRUNTIME_VERSION is not set")
  endif()
  set(ONNXRUNTIME_VER ${OCOS_ONNXRUNTIME_VERSION})
  set(ONNXRUNTIME_URL ${OCOS_ONNXRUNTIME_PKG_URI})
  message(STATUS "ONNX Runtime URL: ${OCOS_ONNXRUNTIME_PKG_URI}")
  FetchContent_Declare(
    onnxruntime
    URL ${OCOS_ONNXRUNTIME_PKG_URI}
  )

  FetchContent_makeAvailable(onnxruntime)

  if (ANDROID)
    set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/headers)
    set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/jni/${ANDROID_ABI})
    message(STATUS "Android onnxruntime inc=${ONNXRUNTIME_INCLUDE_DIR} lib=${ONNXRUNTIME_LIB_DIR}")
  else()
    set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
    set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
  endif()
else()
  message(STATUS "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
  message(STATUS "CMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}")

  # 1.15.1 is the latest ORT release.
  if (OCOS_ONNXRUNTIME_VERSION)
    set(ONNXRUNTIME_VER ${OCOS_ONNXRUNTIME_VERSION})
  else()
    set(ONNXRUNTIME_VER "1.15.1")
  endif()

  if (ANDROID)
    set(ort_fetch_URL "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ONNXRUNTIME_VER}/onnxruntime-android-${ONNXRUNTIME_VER}.aar")
  elseif(IOS)
    set(ort_fetch_URL "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-c-${ONNXRUNTIME_VER}.zip")
  else()
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

    set(ort_fetch_URL "https://github.com/microsoft/onnxruntime/releases/download/${ONNXRUNTIME_URL}")
  endif()

  message(STATUS "ONNX Runtime URL: ${ort_fetch_URL}")
  FetchContent_Declare(
    onnxruntime
    URL ${ort_fetch_URL}
  )

  FetchContent_makeAvailable(onnxruntime)

  if (ANDROID)
    set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/headers)
    set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/jni/${ANDROID_ABI})
    message(STATUS "Android onnxruntime inc=${ONNXRUNTIME_INCLUDE_DIR} lib=${ONNXRUNTIME_LIB_DIR}")
  elseif(IOS)
    set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/Headers)
    # TODO Update once CMake supports finding and linking to .xcframeworks, possibly in 3.28.
    # https://gitlab.kitware.com/cmake/cmake/-/issues/21752
    # https://gitlab.kitware.com/cmake/cmake/-/merge_requests/8619
    # https://gitlab.kitware.com/cmake/cmake/-/merge_requests/8661
    if(CMAKE_OSX_SYSROOT STREQUAL "iphoneos")
      set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/onnxruntime.xcframework/ios-arm64)
    elseif(CMAKE_OSX_SYSROOT STREQUAL "iphonesimulator")
      set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/onnxruntime.xcframework/ios-arm64_x86_64-simulator)
    else()
      message(FATAL_ERROR "Unsupported CMAKE_OSX_SYSROOT value: ${CMAKE_OSX_SYSROOT}")
    endif()
  else()
    set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
    set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
  endif()
endif()

if(NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR})
  message(FATAL_ERROR "ONNX Runtime headers not found at ${ONNXRUNTIME_INCLUDE_DIR}")
endif()
