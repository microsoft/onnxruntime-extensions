# ExternalProject_Add(zlib
#     PREFIX zlib
#     GIT_REPOSITORY https://github.com/madler/zlib.git
#     GIT_TAG v1.2.13
#     GIT_SHALLOW TRUE
# )

# message(STATUS "src:${zlib_SOURCE_DIR} ")


FetchContent_Declare(
    zlib
    GIT_REPOSITORY  "https://github.com/madler/zlib.git"
    GIT_TAG         v1.2.13
    GIT_SHALLOW     TRUE
)

# FetchContent_GetProperties(zlib)
# if(NOT zlib_POPULATED)
#     FetchContent_Populate(zlib)

#     add_subdirectory(${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})

#     target_include_directories(zlib PUBLIC
#         $<BUILD_INTERFACE:${zlib_BINARY_DIR}>
#         $<INSTALL_INTERFACE:include>
#     )

#     target_include_directories(zlibstatic PUBLIC
#         $<BUILD_INTERFACE:${zlib_BINARY_DIR}>
#         $<INSTALL_INTERFACE:include>
#     )
# endif()

FetchContent_MakeAvailable(zlib)

set(zlib_INCLUDE_DIRS ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
set(zlib_LIB_NAME zlibstatic)
