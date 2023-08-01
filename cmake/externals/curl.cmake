if (ANDROID)
  set(PREBUILD_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/prebuild/openssl_for_ios_and_android/output/android")
  set(OPENSSL_ROOT_DIR "${PREBUILD_OUTPUT_PATH}/openssl-${ANDROID_ABI}")
  set(NGHTTP2_ROOT_DIR "${PREBUILD_OUTPUT_PATH}/nghttp2-${ANDROID_ABI}")
  set(CURL_ROOT_DIR "${PREBUILD_OUTPUT_PATH}/curl-${ANDROID_ABI}")

  # Update CMAKE_FIND_ROOT_PATH so find_package/find_library can find these builds
  list(APPEND CMAKE_FIND_ROOT_PATH "${OPENSSL_ROOT_DIR}" )
  list(APPEND CMAKE_FIND_ROOT_PATH "${NGHTTP2_ROOT_DIR}" )
  list(APPEND CMAKE_FIND_ROOT_PATH "${CURL_ROOT_DIR}" )
endif()