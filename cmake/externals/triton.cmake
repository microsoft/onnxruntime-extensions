include(ExternalProject)

set(triton_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton)
set(triton_INSTALL_DIR ${triton_PREFIX}/install)

if (WIN32)
  if (ocos_target_platform STREQUAL "AMD64")
    set(vcpkg_target_platform "x64")
  else()
    set(vcpkg_target_platform ${ocos_target_platform})
  endif()

  set(vcpkg_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/_deps/vcpkg)

  ExternalProject_Add(vcpkg
                      GIT_REPOSITORY https://github.com/microsoft/vcpkg.git
                      GIT_TAG 2023.06.20
                      PREFIX ${vcpkg_PREFIX}
                      CONFIGURE_COMMAND ""
                      INSTALL_COMMAND ""
                      UPDATE_COMMAND ""
                      BUILD_COMMAND "<SOURCE_DIR>/bootstrap-vcpkg.bat")

  ExternalProject_Get_Property(vcpkg SOURCE_DIR BINARY_DIR)
  set(VCPKG_SRC ${SOURCE_DIR})
  message(status "vcpkg source dir: " ${VCPKG_SRC})

  # set the environment variable so that the vcpkg.cmake file can find the vcpkg root directory
  set(ENV{VCPKG_ROOT} ${VCPKG_SRC})

  message(STATUS "VCPKG_SRC: " ${VCPKG_SRC})
  message(STATUS "ENV{VCPKG_ROOT}: " $ENV{VCPKG_ROOT})

  # NOTE: The VCPKG_ROOT environment variable isn't propagated to an add_custom_command target, so specify --vcpkg-root
  # here and in the vcpkg_install function
  add_custom_command(
    COMMAND ${CMAKE_COMMAND} -E echo ${VCPKG_SRC}/vcpkg integrate --vcpkg-root=$ENV{VCPKG_ROOT} install
    COMMAND ${VCPKG_SRC}/vcpkg integrate --vcpkg-root=$ENV{VCPKG_ROOT} install
    COMMAND ${CMAKE_COMMAND} -E touch vcpkg_integrate.stamp
    OUTPUT vcpkg_integrate.stamp
    DEPENDS vcpkg
  )

  add_custom_target(vcpkg_integrate ALL DEPENDS vcpkg_integrate.stamp)
  set(VCPKG_DEPENDENCIES "vcpkg_integrate")

  # use static-md so it adjusts for debug/release CRT
  # https://stackoverflow.com/questions/67258905/vcpkg-difference-between-windows-windows-static-and-other
  function(vcpkg_install PACKAGE_NAME)
    add_custom_command(
      OUTPUT ${VCPKG_SRC}/packages/${PACKAGE_NAME}_${vcpkg_target_platform}-windows-static-md/BUILD_INFO
      COMMAND ${CMAKE_COMMAND} -E echo ${VCPKG_SRC}/vcpkg install --vcpkg-root=$ENV{VCPKG_ROOT}
                                  ${PACKAGE_NAME}:${vcpkg_target_platform}-windows-static-md
      COMMAND ${VCPKG_SRC}/vcpkg install --vcpkg-root=$ENV{VCPKG_ROOT}
                                         ${PACKAGE_NAME}:${vcpkg_target_platform}-windows-static-md
      WORKING_DIRECTORY ${VCPKG_SRC}
      DEPENDS vcpkg_integrate)

    add_custom_target(
      get${PACKAGE_NAME}
      ALL
      DEPENDS ${VCPKG_SRC}/packages/${PACKAGE_NAME}_${vcpkg_target_platform}-windows-static-md/BUILD_INFO)

    list(APPEND VCPKG_DEPENDENCIES "get${PACKAGE_NAME}")
    set(VCPKG_DEPENDENCIES ${VCPKG_DEPENDENCIES} PARENT_SCOPE)
  endfunction()

  vcpkg_install(rapidjson)
  vcpkg_install(openssl)
  vcpkg_install(curl)

  set(triton_extra_cmake_args -DVCPKG_TARGET_TRIPLET=${vcpkg_target_platform}-windows-static-md
                              -DCMAKE_TOOLCHAIN_FILE=${VCPKG_SRC}/scripts/buildsystems/vcpkg.cmake)
  set(triton_dependencies ${VCPKG_DEPENDENCIES})

else()
  # RapidJSON 1.1.0 (released in 2016) is compatible with the triton build. Later code is not compatible without
  # patching due to the change in variable name for the include dir from RAPIDJSON_INCLUDE_DIRS to
  # RapidJSON_INCLUDE_DIRS in the generated cmake file used by find_package:
  #   https://github.com/Tencent/rapidjson/commit/b91c515afea9f0ba6a81fc670889549d77c83db3
  # The triton code here https://github.com/triton-inference-server/common/blob/main/CMakeLists.txt is using
  # RAPIDJSON_INCLUDE_DIRS so the build fails if a newer RapidJSON version is used. It will find the package but the
  # include path will be wrong so the build error is delayed/misleading and non-trivial to understand/resolve.
  set(RapidJSON_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/_deps/rapidjson)
  set(RapidJSON_INSTALL_DIR ${RapidJSON_PREFIX}/install)
  ExternalProject_Add(RapidJSON
                      PREFIX ${RapidJSON_PREFIX}
                      URL https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.zip
                      URL_HASH SHA1=0fe7b4f7b83df4b3d517f4a202f3a383af7a0818
                      CMAKE_ARGS -DRAPIDJSON_BUILD_DOC=OFF
                                 -DRAPIDJSON_BUILD_EXAMPLES=OFF
                                 -DRAPIDJSON_BUILD_TESTS=OFF
                                 -DRAPIDJSON_HAS_STDSTRING=ON
                                 -DRAPIDJSON_USE_MEMBERSMAP=ON
                                 -DCMAKE_INSTALL_PREFIX=${RapidJSON_INSTALL_DIR}
                                 )

  ExternalProject_Get_Property(RapidJSON SOURCE_DIR BINARY_DIR)
  # message(STATUS "RapidJSON src=${SOURCE_DIR} binary=${BINARY_DIR}")
  # Set RapidJSON_ROOT_DIR for find_package. The required RapidJSONConfig.cmake file is generated in the binary dir
  set(RapidJSON_ROOT_DIR ${BINARY_DIR})

  set(triton_extra_cmake_args "")
  set(triton_dependencies RapidJSON)
endif() #if (WIN32)

# Add the triton build. We just need the library so we don't install it.
#
# Patch the triton client CMakeLists.txt to fix two issues when building the python wheels with cibuildwheel, which
# uses CentOS 7.
#  1) use the full path to the version script file so 'ld' doesn't fail to find it. Looks like ld is running from the 
#     parent directory but not sure why the behavior differs vs. other linux builds 
#     e.g. building locally on Ubuntu is fine without the patch
#  2) only set the CURL lib path to 'lib64' on a 64-bit CentOS build as 'lib64' is invalid on a 32-bit OS. without
#     this patch the build of the third-party libraries in the triton client fail as the CURL build is not found.
ExternalProject_Add(triton
                    URL https://github.com/triton-inference-server/client/archive/refs/heads/r23.05.tar.gz
                    URL_HASH SHA1=b8fd2a4e09eae39c33cd04cfa9ec934e39d9afc1
                    PREFIX ${triton_PREFIX}
                    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${triton_INSTALL_DIR}
                               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                               -DTRITON_ENABLE_CC_HTTP=ON
                               -DTRITON_ENABLE_ZLIB=OFF
                               ${triton_extra_cmake_args}                    
                    INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Skipping install step."
                    PATCH_COMMAND patch --verbose -p1 -i ${PROJECT_SOURCE_DIR}/cmake/externals/triton_cmake.patch
                    )

add_dependencies(triton ${triton_dependencies})

ExternalProject_Get_Property(triton SOURCE_DIR BINARY_DIR)
set(triton_THIRD_PARTY_DIR ${BINARY_DIR}/third-party)

