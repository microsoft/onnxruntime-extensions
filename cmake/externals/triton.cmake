include(ExternalProject)

if (WIN32)

  if (ocos_target_platform STREQUAL "AMD64")
    set(vcpkg_target_platform "x86")
  else()
    set(vcpkg_target_platform ${ocos_target_platform})
  endif()

  ExternalProject_Add(vcpkg
    GIT_REPOSITORY https://github.com/microsoft/vcpkg.git
    GIT_TAG 2023.06.20
    PREFIX vcpkg
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/vcpkg-src
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/vcpkg-build
    CONFIGURE_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    BUILD_COMMAND "<SOURCE_DIR>/bootstrap-vcpkg.bat")

  set(VCPKG_SRC ${CMAKE_CURRENT_BINARY_DIR}/_deps/vcpkg-src)
  set(ENV{VCPKG_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/_deps/vcpkg-src)

  message(STATUS "VCPKG_SRC: " ${VCPKG_SRC})
  message(STATUS "VCPKG_ROOT: " $ENV{VCPKG_ROOT})

  add_custom_command(
    COMMAND ${VCPKG_SRC}/vcpkg integrate install
    COMMAND ${CMAKE_COMMAND} -E touch vcpkg_integrate.stamp
    OUTPUT vcpkg_integrate.stamp
    DEPENDS vcpkg
  )

  add_custom_target(vcpkg_integrate ALL DEPENDS vcpkg_integrate.stamp)
  set(VCPKG_DEPENDENCIES "vcpkg_integrate")

  function(vcpkg_install PACKAGE_NAME)
    add_custom_command(
      OUTPUT ${VCPKG_SRC}/packages/${PACKAGE_NAME}_${vcpkg_target_platform}-windows-static/BUILD_INFO
      COMMAND ${VCPKG_SRC}/vcpkg install ${PACKAGE_NAME}:${vcpkg_target_platform}-windows-static --vcpkg-root=${CMAKE_CURRENT_BINARY_DIR}/_deps/vcpkg-src
      WORKING_DIRECTORY ${VCPKG_SRC}
      DEPENDS vcpkg_integrate)

    add_custom_target(get${PACKAGE_NAME}
      ALL
      DEPENDS ${VCPKG_SRC}/packages/${PACKAGE_NAME}_${vcpkg_target_platform}-windows-static/BUILD_INFO)

    list(APPEND VCPKG_DEPENDENCIES "get${PACKAGE_NAME}")
    set(VCPKG_DEPENDENCIES ${VCPKG_DEPENDENCIES} PARENT_SCOPE)
  endfunction()

  vcpkg_install(openssl)
  vcpkg_install(openssl-windows)
  vcpkg_install(rapidjson)
  vcpkg_install(re2)
  vcpkg_install(boost-interprocess)
  vcpkg_install(boost-stacktrace)
  vcpkg_install(pthread)
  vcpkg_install(b64)

  add_dependencies(getb64 getpthread)
  add_dependencies(getpthread getboost-stacktrace)
  add_dependencies(getboost-stacktrace getboost-interprocess)
  add_dependencies(getboost-interprocess getre2)
  add_dependencies(getre2 getrapidjson)
  add_dependencies(getrapidjson getopenssl-windows)
  add_dependencies(getopenssl-windows getopenssl)

  ExternalProject_Add(triton
                      GIT_REPOSITORY https://github.com/triton-inference-server/client.git
                      GIT_TAG r23.05
                      PREFIX triton
                      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton-src
                      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton-build
                      CMAKE_ARGS -DVCPKG_TARGET_TRIPLET=${vcpkg_target_platform}-windows-static -DCMAKE_TOOLCHAIN_FILE=${VCPKG_SRC}/scripts/buildsystems/vcpkg.cmake -DCMAKE_INSTALL_PREFIX=binary -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_ZLIB=OFF
                      INSTALL_COMMAND ""
                      UPDATE_COMMAND "")

  add_dependencies(triton ${VCPKG_DEPENDENCIES})

else()

  if(DEFINED ENV{IS_DOCKER_BUILD})
    ExternalProject_Add(curl7
                        PREFIX curl7
                        GIT_REPOSITORY "https://github.com/curl/curl.git"
                        GIT_TAG "curl-7_86_0"
                        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/curl7-src
                        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/curl7-build
                        CMAKE_ARGS -DBUILD_TESTING=OFF -DBUILD_CURL_EXE=OFF -DBUILD_SHARED_LIBS=OFF -DCURL_STATICLIB=ON -DHTTP_ONLY=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
  endif()

  ExternalProject_Add(triton
                      GIT_REPOSITORY https://github.com/triton-inference-server/client.git
                      GIT_TAG r23.05
                      PREFIX triton
                      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton-src
                      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton-build
                      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=binary -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_ZLIB=OFF
                      INSTALL_COMMAND ""
                      UPDATE_COMMAND "")

  add_dependencies(triton curl7)

endif() #if (WIN32)

ExternalProject_Get_Property(triton SOURCE_DIR)
set(TRITON_SRC ${SOURCE_DIR})

ExternalProject_Get_Property(triton BINARY_DIR)
set(TRITON_BIN ${BINARY_DIR}/binary)
set(TRITON_THIRD_PARTY ${BINARY_DIR}/third-party)
