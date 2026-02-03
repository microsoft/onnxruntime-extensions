include(ExternalProject)

set(triton_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton)
set(triton_INSTALL_DIR ${triton_PREFIX}/install)

if (WIN32)
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
  message(STATUS "VCPKG_SRC: " ${VCPKG_SRC})

  # set the environment variable so that the vcpkg.cmake file can find the vcpkg root directory
  set(ENV{VCPKG_ROOT} ${VCPKG_SRC})
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

  # we need the installs to be sequential otherwise you get strange build errors,
  # and create fake dependencies between each to make the install of each package sequential
  set(PACKAGE_NAMES rapidjson  # required by triton
                    zlib       # required by curl. zlib also comes from opencv if enabled. TODO: check compatibility
                    openssl
                    curl)

  foreach(PACKAGE_NAME ${PACKAGE_NAMES})
    message(STATUS "Adding vcpkg package: ${PACKAGE_NAME}")
    add_custom_command(
      OUTPUT ${VCPKG_SRC}/packages/${PACKAGE_NAME}_${vcpkg_triplet}/BUILD_INFO
      COMMAND ${CMAKE_COMMAND} -E echo ${VCPKG_SRC}/vcpkg install --vcpkg-root=$ENV{VCPKG_ROOT}
                                  ${PACKAGE_NAME}:${vcpkg_triplet}
      COMMAND ${VCPKG_SRC}/vcpkg install --vcpkg-root=$ENV{VCPKG_ROOT}
                                         ${PACKAGE_NAME}:${vcpkg_triplet}
      WORKING_DIRECTORY ${VCPKG_SRC}
      DEPENDS vcpkg_integrate)

    add_custom_target(
      get${PACKAGE_NAME}
      ALL
      DEPENDS ${VCPKG_SRC}/packages/${PACKAGE_NAME}_${vcpkg_triplet}/BUILD_INFO)

    set(_cur_package "get${PACKAGE_NAME}")
    list(APPEND VCPKG_DEPENDENCIES ${_cur_package})

    # chain the dependencies so that the packages are installed sequentially by cmake
    if(_prev_package)
      add_dependencies(${_cur_package} ${_prev_package})
    endif()
    set(_prev_package ${_cur_package})
  endforeach()

  set(triton_extra_cmake_args -DVCPKG_TARGET_TRIPLET=${vcpkg_triplet}
                              -DCMAKE_TOOLCHAIN_FILE=${VCPKG_SRC}/scripts/buildsystems/vcpkg.cmake)

  # insane... but we added patch.exe from the git 32-bit windows distribution because the CI image has removed it
  # from the git install that is available, and this seems like the least egregious/inconsistent way to make it work...
  set(triton_patch_exe ${PROJECT_SOURCE_DIR}/cmake/externals/git.Win32.2.41.03.patch/patch.exe)
  set(triton_dependencies ${VCPKG_DEPENDENCIES})
else()
  # RapidJSON 1.1.0 (released in 2016) has C++ compatibility issues with GCC 14 (deprecated std::iterator,
  # const member assignment). Use master branch which includes these fixes and already has cmake_minimum_required 3.5.
  #
  # Note: Newer RapidJSON uses modern CMake with exported targets and changed the variable name from
  # RAPIDJSON_INCLUDE_DIRS to RapidJSON_INCLUDE_DIRS. Triton expects the uppercase version, so we pass
  # RAPIDJSON_INCLUDE_DIRS directly as a cmake argument to the triton build.
  set(RapidJSON_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/_deps/rapidjson)
  set(RapidJSON_INSTALL_DIR ${RapidJSON_PREFIX}/install)
  ExternalProject_Add(RapidJSON
                      PREFIX ${RapidJSON_PREFIX}
                      GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
                      GIT_TAG master
                      GIT_SHALLOW TRUE
                      CMAKE_ARGS -DRAPIDJSON_BUILD_DOC=OFF
                                 -DRAPIDJSON_BUILD_EXAMPLES=OFF
                                 -DRAPIDJSON_BUILD_TESTS=OFF
                                 -DRAPIDJSON_HAS_STDSTRING=ON
                                 -DRAPIDJSON_USE_MEMBERSMAP=ON
                                 -DCMAKE_INSTALL_PREFIX=${RapidJSON_INSTALL_DIR}
                                 )

  ExternalProject_Get_Property(RapidJSON SOURCE_DIR BINARY_DIR)
  # message(STATUS "RapidJSON src=${SOURCE_DIR} binary=${BINARY_DIR}")
  # Set RapidJSON_ROOT_DIR for find_package. The cmake config files are installed to lib/cmake/RapidJSON
  set(RapidJSON_ROOT_DIR ${RapidJSON_INSTALL_DIR}/lib/cmake/RapidJSON)
  # Triton expects RAPIDJSON_INCLUDE_DIRS (uppercase). Pass it directly as a cmake arg.
  set(triton_extra_cmake_args -DRAPIDJSON_INCLUDE_DIRS=${RapidJSON_INSTALL_DIR}/include)
  set(triton_patch_exe patch)
  set(triton_dependencies RapidJSON)

endif() #if (WIN32)

# Add the triton build. We just need the library so we don't install it.
set(triton_VERSION_TAG r23.05)

# Patch the triton client CMakeLists.txt to fix issues when building the linux python wheels with cibuildwheel, which
# uses CentOS 7.
#  1) use the full path to the version script file so 'ld' doesn't fail to find it. Looks like ld is running from the
#     parent directory but not sure why the behavior differs vs. other linux builds
#       e.g. building locally on Ubuntu is fine without the patch
#  2) only set the CURL lib path to 'lib64' on a 64-bit CentOS build as 'lib64' is invalid on a 32-bit OS. without
#     this patch the build of the third-party libraries in the triton client fail as the CURL build is not found.
#
# Also fix a Windows issue where the security tools complain about warnings being disabled by increasing the
# warning level for triton/src/c++/library. Was /W0. Patched to /W3.
set(triton_patch_command ${triton_patch_exe} --verbose -p1 -i ${PROJECT_SOURCE_DIR}/cmake/externals/triton_cmake.patch)

ExternalProject_Add(triton
                    URL https://github.com/triton-inference-server/client/archive/refs/heads/${triton_VERSION_TAG}.tar.gz
                    URL_HASH SHA1=b8fd2a4e09eae39c33cd04cfa9ec934e39d9afc1
                    PREFIX ${triton_PREFIX}
                    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton-src
                    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton-build
                    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${triton_INSTALL_DIR}
                               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                               -DTRITON_COMMON_REPO_TAG=${triton_VERSION_TAG}
                               -DTRITON_THIRD_PARTY_REPO_TAG=${triton_VERSION_TAG}
                               -DTRITON_CORE_REPO_TAG=${triton_VERSION_TAG}
                               -DTRITON_ENABLE_CC_HTTP=ON
                               -DTRITON_ENABLE_ZLIB=OFF
                               ${triton_extra_cmake_args}
                    INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Skipping install step."
                    PATCH_COMMAND ${triton_patch_command}
                    )

add_dependencies(triton ${triton_dependencies})

ExternalProject_Get_Property(triton SOURCE_DIR BINARY_DIR)
set(triton_THIRD_PARTY_DIR ${BINARY_DIR}/third-party)
