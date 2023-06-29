include(ExternalProject)

if (WIN32)

  if (ocos_target_platform STREQUAL "AMD64")
    set(vcpkg_target_platform "x64")
  else()
    set(vcpkg_target_platform ${ocos_target_platform})
  endif()

  set(VCPKG_SRC $ENV{VCPKG_ROOT})
  message(WARNING "VCPKG_SRC: " ${VCPKG_SRC})
  message(WARNING "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})

  add_custom_command(
    COMMAND ${VCPKG_SRC}/vcpkg install
    COMMAND ${CMAKE_COMMAND} -E touch vcpkg_install.stamp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT vcpkg_install.stamp
  )

  add_custom_target(vcpkg_install ALL DEPENDS vcpkg_install.stamp)

  ExternalProject_Add(triton
                      GIT_REPOSITORY https://github.com/triton-inference-server/client.git
                      GIT_TAG r23.05
                      PREFIX triton
                      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton-src
                      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton-build
                      CMAKE_ARGS -DVCPKG_TARGET_TRIPLET=${vcpkg_target_platform}-windows-static -DCMAKE_TOOLCHAIN_FILE=${VCPKG_SRC}/scripts/buildsystems/vcpkg.cmake -DCMAKE_INSTALL_PREFIX=binary -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_ZLIB=OFF
                      INSTALL_COMMAND ""
                      UPDATE_COMMAND "")

  add_dependencies(triton vcpkg_install)

else()

  ExternalProject_Add(triton
                      GIT_REPOSITORY https://github.com/triton-inference-server/client.git
                      GIT_TAG r23.05
                      PREFIX triton
                      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton-src
                      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/triton-build
                      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=binary -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_ZLIB=OFF
                      INSTALL_COMMAND ""
                      UPDATE_COMMAND "")

endif() #if (WIN32)

ExternalProject_Get_Property(triton SOURCE_DIR)
set(TRITON_SRC ${SOURCE_DIR})

ExternalProject_Get_Property(triton BINARY_DIR)
set(TRITON_BIN ${BINARY_DIR}/binary)
set(TRITON_THIRD_PARTY ${BINARY_DIR}/third-party)
