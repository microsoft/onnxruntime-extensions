# ----------------------------------------------------------------------------
#  CMake file for libjpeg. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
project(JPEG_LIBRARY)

file(GLOB lib_srcs *.c)
file(GLOB lib_hdrs *.h)

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

add_library(JPEG_LIBRARY STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${lib_srcs} ${lib_hdrs})

if(CV_GCC OR CV_CLANG)
  set_source_files_properties(jcdctmgr.c PROPERTIES COMPILE_FLAGS "-O1")
endif()

set_target_properties(JPEG_LIBRARY
  PROPERTIES OUTPUT_NAME JPEG_LIBRARY
  DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
  COMPILE_PDB_NAME JPEG_LIBRARY
  COMPILE_PDB_NAME_DEBUG "JPEG_LIBRARY${OPENCV_DEBUG_POSTFIX}"
  ARCHIVE_OUTPUT_DIRECTORY "${3P_LIBRARY_OUTPUT_PATH}"
  )

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(JPEG_LIBRARY PROPERTIES FOLDER "3rdparty")
endif()
