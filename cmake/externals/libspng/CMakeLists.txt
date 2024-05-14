# ----------------------------------------------------------------------------
#  CMake file for libspng. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

project(SPNG)

set(CURR_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set_property(GLOBAL PROPERTY SPNG_INCLUDE_DIR ${CURR_INCLUDE_DIR})

file(GLOB_RECURSE spng_headers RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "*.h")
file(GLOB_RECURSE spng_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "*.c")

message(STATUS "libspng will be used as PNG codec")

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif(MSVC)

add_library(SPNG STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${spng_headers} ${spng_sources})
target_link_libraries(SPNG ${ZLIB_LIBRARIES})

set_target_properties(SPNG
        PROPERTIES OUTPUT_NAME SPNG
        DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
        COMPILE_PDB_NAME SPNG
        COMPILE_PDB_NAME_DEBUG "SPNG${OPENCV_DEBUG_POSTFIX}"
        ARCHIVE_OUTPUT_DIRECTORY "${3P_LIBRARY_OUTPUT_PATH}"
        )

target_compile_definitions(SPNG PUBLIC SPNG_STATIC)

if(ENABLE_SOLUTION_FOLDERS)
    set_target_properties(SPNG PROPERTIES FOLDER "3rdparty")
endif()