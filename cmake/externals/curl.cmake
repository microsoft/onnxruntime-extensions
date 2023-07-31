if (ANDROID)
  set(PREBUILD_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/prebuild/openssl_for_ios_and_android/output/android")
  set(OPENSSL_ROOT_DIR "${PREBUILD_OUTPUT_PATH}/openssl-${ANDROID_ABI}")
  set(NGHTTP2_ROOT_DIR "${PREBUILD_OUTPUT_PATH}/nghttp2-${ANDROID_ABI}")
  set(CURL_ROOT_DIR "${PREBUILD_OUTPUT_PATH}/curl-${ANDROID_ABI}")

  # Add paths so find_package works for the libraries
  # TODO: If we use <PackageName>_ROOT and global scope can we avoid adding to CMAKE_FIND_ROOT_PATH?
  set(OPENSSL_ROOT "${PREBUILD_OUTPUT_PATH}/openssl-${ANDROID_ABI}" PARENT_SCOPE)
  set(NGHTTP2_ROOT "${PREBUILD_OUTPUT_PATH}/nghttp2-${ANDROID_ABI}" PARENT_SCOPE)
  set(CURL_ROOT "${PREBUILD_OUTPUT_PATH}/curl-${ANDROID_ABI}" PARENT_SCOPE)

  # list(APPEND CMAKE_FIND_ROOT_PATH "${OPENSSL_ROOT_DIR}" )
  # list(APPEND CMAKE_FIND_ROOT_PATH "${NGHTTP2_ROOT_DIR}" )
  # list(APPEND CMAKE_FIND_ROOT_PATH "${CURL_ROOT_DIR}" )
endif()