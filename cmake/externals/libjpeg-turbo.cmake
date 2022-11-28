
# set(ENABLE_SHARED OFF CACHE INTERNAL "")

# FetchContent_Declare(
#   libjpeg-turbo
#   GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo.git
#   GIT_TAG        2.1.4
#   # GIT_SHALLOW    TRUE
#   PATCH_COMMAND git checkout . && git apply --whitespace=fix --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_SOURCE_DIR}/cmake/externals/libjpeg-turbo.patch
# )

# FetchContent_MakeAvailable(libjpeg-turbo)

if (NOT OCOS_BUILD_ANDROID)
    ExternalProject_Add(libjpeg-turbo
        GIT_REPOSITORY  https://github.com/libjpeg-turbo/libjpeg-turbo.git
        GIT_TAG         2.1.4
        GIT_SHALLOW     TRUE
        PREFIX          ${CMAKE_CURRENT_BINARY_DIR}/_deps/libjpeg-turbo
        INSTALL_COMMAND cmake -E echo "Skipping install step for dependency libjpeg-turbo"
        INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    )
else()
    message(STATUS "Toolchain: ${CMAKE_TOOLCHAIN_FILE}")

    ExternalProject_Add(libjpeg-turbo
        GIT_REPOSITORY  https://github.com/libjpeg-turbo/libjpeg-turbo.git
        GIT_TAG         2.1.4
        GIT_SHALLOW     TRUE
        PREFIX          ${CMAKE_CURRENT_BINARY_DIR}/_deps/libjpeg-turbo
        INSTALL_COMMAND cmake -E echo "Skipping install step for dependency libjpeg-turbo"
        INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
        CMAKE_ARGS
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DANDROID_ABI=${ANDROID_ABI}
            -DANDROID_PLATFORM=${ANDROID_PLATFORM}
    )
endif()

set (libjpeg-turbo_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/libjpeg-turbo/src/libjpeg-turbo)
set (libjpeg-turbo_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/libjpeg-turbo/src/libjpeg-turbo-build)

if (MSVC)
    link_directories(${libjpeg-turbo_BINARY_DIR}/${CMAKE_BUILD_TYPE})
else()
    link_directories(${libjpeg-turbo_BINARY_DIR})
endif()
