
# set(ENABLE_SHARED OFF CACHE INTERNAL "")

# FetchContent_Declare(
#   libjpeg-turbo
#   GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo.git
#   GIT_TAG        2.1.4
#   # GIT_SHALLOW    TRUE
#   PATCH_COMMAND git checkout . && git apply --whitespace=fix --ignore-space-change --ignore-whitespace ${CMAKE_CURRENT_SOURCE_DIR}/cmake/externals/libjpeg-turbo.patch
# )

# FetchContent_MakeAvailable(libjpeg-turbo)

ExternalProject_Add(libjpeg-turbo
    GIT_REPOSITORY  https://github.com/libjpeg-turbo/libjpeg-turbo.git
    GIT_TAG         2.1.4
    GIT_SHALLOW     TRUE
    PREFIX          _deps/libjpeg-turbo
    INSTALL_COMMAND cmake -E echo "Skipping install step for dependency libjpeg-turbo"
    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
)

set (libjpeg-turbo_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/libjpeg-turbo/src)
# D:\src\github\ort-extensions\build\Windows\Debug\_deps\libjpeg-turbo\src\libjpeg-turbo
